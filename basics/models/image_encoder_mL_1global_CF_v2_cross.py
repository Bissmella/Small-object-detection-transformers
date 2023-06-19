import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Tuple, Type
from functools import partial
from .SAM_commons import MLPBlock, LayerNorm2d

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 4,
        embed_dim: int = 768,
        depth: int = 11,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6), # nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,        #False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (), #*changed
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(1, 1),  #patch_size
            stride=(1, 1),         #* previoulsy 4  changed from patch_size, patch_size to 8, 8 to get half-overlapping 64 x 64 dimension patches
            padding = (0, 0),
            in_chans=192,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // 4, img_size // 4, embed_dim)   #*changed from patch_size to 4   img_size // 4 changed to 160
            )

        #for channel attention
        self.channel_embed_r = PatchEmbed(
            kernel_size = (patch_size, patch_size),
            stride = (4, 4),
            in_chans = 1,
            embed_dim = 48,
        )

        self.channel_embed_g = PatchEmbed(
            kernel_size = (patch_size, patch_size),
            stride = (4, 4),
            in_chans = 1,
            embed_dim = 48,
        )

        self.channel_embed_b = PatchEmbed(
            kernel_size = (patch_size, patch_size),
            stride = (4, 4),
            in_chans = 1,
            embed_dim = 48,
        )

        self.channel_embed_i = PatchEmbed(
            kernel_size = (patch_size, patch_size),
            stride = (4, 4),
            in_chans = 1,
            embed_dim = 48,
        )
        
        self.chan_block = CAttentionBlock(
                embedding_dim = 48,
                num_heads = num_heads,
            )
            

        self.fc_layer = MLPBlock(embedding_dim=192, mlp_dim=384, act=nn.GELU)
        #nn.Linear(192, 192) #removed (img_size //16) *
        #local blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // 4, img_size // 4),   #* changed from  // patch_size to 4
            )
            self.blocks.append(block)
        #a second pathc embedding
        self.patch_embed2 = PatchEmbed(
            kernel_size=(4, 4),
            stride=(4,4),
            padding=(0, 0),
            in_chans=192,
            embed_dim=768,
        )
        #global attention module
        self.glob_block = Block(
            dim = 768,
            num_heads = num_heads,
            mlp_ratio= mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=0,
            input_size=(128 // 4, 128 // 4),
        )
        self.neck3 = nn.Sequential(
            nn.Conv2d(
                768,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.neck2 = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.neck1 = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
    '''
    def fuse_chan(self, x: torch.Tensor):
        if x.shape[2] != 512:
            pad_height = max(512 - x.shape[2], 0)
            pad_width = max(512 - x.shape[3], 0)

            # Pad the image with zeros
            x = F.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0)
        bs, h, w, c = x.shape
        r = self.channel_embed(x[:,0,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        g = self.channel_embed(x[:,1,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        b = self.channel_embed(x[:,2,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        i = self.channel_embed(x[:,3,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)

        chans = torch.cat((r, g, b, i), dim=1)

        chans = self.chan_block(chans)
        bs, h, w, c     = chans.shape

        blocks = self.img_size // 16

        bs = chans.shape[0]
        chans_flat = chans.view(bs, -1)
        chans_out = self.fc_layer(chans_flat)
        chans_out = chans_out.view(bs, 1, 192)
        chans_out = chans_out.repeat(1, blocks, 1)
        self.chan_atten = chans_out
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #fuse channels:
        
        ##patches = self.patch_embed(x)
        #bs, h, w, c = x.shape
        r, g ,b, i = get_channels(x)
        r = self.channel_embed_r(r)#.unsqueeze(1)      #x1[:,0,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        g = self.channel_embed_g(g)#.unsqueeze(1)      #x1[:,1,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        b = self.channel_embed_b(b)#.unsqueeze(1)      #x1[:,2,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        i = self.channel_embed_i(i)#.unsqueeze(1)       #x1[:,3,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)

        # x = torch.cat((r, g, b, i), 1)
        # bs, c , h, w, emb = x.shape
        # x = x.reshape(bs, h * w, c, emb)
        # x = channel_partition(x)

        # for blk in self.chan_block:
        #     x = blk(x)
        # x = channel_unpartition(x, h, w, c)
        # x = x.view(bs, h, w, 1, -1).squeeze(3)
        # x = self.fc_layer(x)

        x = self.chan_block(r, g, b, i)
        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed(x)

        y = []
        if self.pos_embed is not None:
            if x.shape[1] == self.pos_embed.shape[1]: #patches before
                x = x + self.pos_embed
        #x = x1 + patches

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i in (9, 10):
                y.append(x)
        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed2(x)
        x = self.glob_block(x)
        y.append(x)
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.neck(x.permute(0, 3, 1, 2))
        W = y[0].shape[2]
        Wg = x.shape[2]
        #y[0] = y[0] + x
        y[0] = self.neck1(y[0].permute(0, 3, 1, 2)) ##[:, :, torch.arange(W) % 5 != 4,:]
        y[1] = self.neck2(y[1].permute(0, 3, 1, 2)) ##[:, :, torch.arange(W) % 5 != 4,:]
        y[2] =  F.interpolate(self.neck3(y[2].permute(0, 3, 1, 2)), scale_factor=4, mode='bilinear', align_corners=False)  ##[:, :, torch.arange(Wg) % 5 != 4,:]
        return y

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class CAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        out_dim: int = 192,
        activation: Type[nn.Module] = nn.ReLU,
        skip_pe: bool = True,
        ) -> None:
        '''
        transformer block for calculating intra channel attention
        '''

        super().__init__()


        self.r2g_attn = CAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.rg2b_attn = CAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)


        self.rgb2ir_attn = CAttention(embedding_dim, num_heads)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.ir2rgb_attn = CAttention(embedding_dim, num_heads)
        self.norm4 = nn.LayerNorm(embedding_dim)
        
        # self.fc_layer = nn.Linear (192, out_dim)
        # #self.dropout = nn.Dropout(0.4)
        # self.mlp = MLPBlock(out_dim, 256, activation)
        # self.norm5 = nn.LayerNorm(out_dim)

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, ir: torch.Tensor):
        b1, h, w, c = r.shape
        r, r_hw =window_partition(r, 2)
        g, g_hw = window_partition(g, 2)
        b, b_hw = window_partition(b, 2)
        ir, ir_hw = window_partition(ir, 2)
        b2, h2, w2, c2 = r.shape
        r = r.reshape(b2, h2 * w2, c2)
        g = g.reshape(b2, h2 * w2, c2)
        b = b.reshape(b2, h2 * w2, c2)
        ir = ir.reshape(b2, h2 * w2, c2)

        attn_out = self.r2g_attn(q = r, k =g, v =g)
        x1 = r + attn_out
        x1 = self.norm1(x1)

        attn_out = self.rg2b_attn(q = g, k =b, v =b)
        x2 = g + attn_out
        x2 = self.norm2(x2)


        attn_out = self.rgb2ir_attn(q =b, k =ir, v =ir)
        x3 = b + attn_out
        x3 = self.norm3(x3)

        attn_out = self.ir2rgb_attn(q = ir, k =g, v =g )
        x4 = ir + attn_out
        x4 = self.norm4(x4)


        x1 = x1.view(b2, h2, w2, c2)
        x2 = x2.view(b2, h2, w2, c2)
        x3 = x3.view(b2, h2, w2, c2)
        x4 = x4.view(b2, h2, w2, c2)

        x1 = window_unpartition(x1, 2, r_hw, (h, w))
        x2 = window_unpartition(x2, 2, g_hw, (h, w))
        x3 = window_unpartition(x3, 2, b_hw, (h, w))
        x4 = window_unpartition(x4, 2, ir_hw, (h, w))
        x = torch.cat((x1, x2, x3, x4), dim=-1)
        # x = self.fc_layer(x)
        # #x = self.dropout(x)
        # x = self.mlp(x)
        # x = self.norm5(x)
        return x

class CAttention(nn.Module):
    """attention layer allowing cross attention used for channels"""

    def __init__(self, 
                embedding_dim:int,
                num_heads: int= 8,
                ) -> None:
                super().__init__()
                self.embedding_dim = embedding_dim
                self.num_heads = num_heads

                self.v_proj = nn.Linear(embedding_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        #Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        #output
        out = attn @ v
        out = self._recombine_heads(out)
        return out


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    ) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),                         #* changed stride from 16, 16  to 8 x 8
        padding: Tuple[int, int] = (1, 1),                          #* originally 0, 0padding should be 4, 4 in case stride is 8 , 8
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
    

class ChanEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self):
        super(ChanEmbed, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2)
        self.c1d = nn.Conv1d(16, 1, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1536)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), 16,  -1)
        x = self.c1d(x)
        x = self.pool(x)
        return x


def channel_partition(x):
    x = x.squeeze(1)
    bs, h, w, e = x.shape
    x = x.contiguous().view(-1, 1, e)
    return x

def channel_unpartition(x, h, w):
    B = x.shape[0] // (h * w)
    x = x.view(B, h, w, -1)
    return x

def get_channels(x):
    #idx = torch.tensor([0]).to(x.device)
    r = x[:,0,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,0,:,:]
    #idx = torch.tensor([1]).to(x.device)
    g = x[:,1,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,1,:,:].unsqueeze(1)
    #idx = torch.tensor([2]).to(x.device)
    b = x[:,2,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,2,:,:].unsqueeze(1)
    #idx = torch.tensor([3]).to(x.device)
    i = x[:,3,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,3,:,:].unsqueeze(1)
    return r, g, b, i


