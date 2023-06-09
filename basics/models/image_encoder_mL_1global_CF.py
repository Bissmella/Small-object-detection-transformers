import torch
import torch.nn as nn
import torch.nn.functional as F

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
            kernel_size=(patch_size, patch_size),
            stride=(4, 4),         #*changed from patch_size, patch_size to 8, 8 to get half-overlapping 64 x 64 dimension patches
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // 4, 160, embed_dim)   #*changed from patch_size to 4   img_size // 4 changed to 160
            )

        #for channel attention
        self.channel_embed = ChanEmbed()
        
        self.chan_block = Block(
            dim = 1536,
            num_heads = num_heads,
            mlp_ratio= mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=0,
            input_size=(1, 1),
        )

        self.fc_layer = nn.Linear(in_chans * 1536, 192) #removed (img_size //16) *
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
        
        if x.shape[2] != 512:
            pad_height = max(512 - x.shape[2], 0)
            pad_width = max(512 - x.shape[3], 0)

            # Pad the image with zeros
            x1 = F.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0)
        else: x1 = x
        bs, h, w, c = x.shape
        r = self.channel_embed(x1[:,0,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        g = self.channel_embed(x1[:,1,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        b = self.channel_embed(x1[:,2,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        i = self.channel_embed(x1[:,3,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)

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
        #the rest
        #self.fuse_chan(x)
        x = self.patch_embed(x)
        bs, h, w, c = x.shape
        self.chan_atten = self.chan_atten.unsqueeze(1).repeat(1, h, 1, 1)   #on the height dimension of the channel attention we repeat as the number of rows of image patches, so the channel attention is for 32 columns of image and only devided on 1 dimension, for both dimension it would be bigger and expensive
        x = x.reshape(bs, h, -1, 4, 192)  # reshaping to the block size, as later the patches will be devided to 4 x 4 blocks, so in the width dimension after each 4 we add one channel dimension
        cb, ch, cblock, cc = self.chan_atten.shape
        self.chan_atten = self.chan_atten.view(bs, h, cblock, -1, cc)  #getting 32 or 16(in case of 256 size of demo image) blocks with 1 x 192 embedding
        #self.chan_atten = self.chan_atten[:, :, :x.shape[2], :, :]   #superyolo passes a demo image of size 256 to specify the stride, so then the number of blocks will 16 and from the channel attention we take only the first 16
        if x.shape[2] == self.chan_atten.shape[2]:
            x = torch.cat((x, self.chan_atten), dim=3)
        x = x.reshape(bs, h, -1, c)
        y = []
        if self.pos_embed is not None:
            if x.shape[1] == self.pos_embed.shape[1]:
                x = x + self.pos_embed
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
        y[0] = self.neck1(y[0][:, :, torch.arange(W) % 5 != 4,:].permute(0, 3, 1, 2)) ##[:, :, torch.arange(W) % 5 != 4,:]
        y[1] = self.neck2(y[1][:, :, torch.arange(W) % 5 != 4,:].permute(0, 3, 1, 2)) ##[:, :, torch.arange(W) % 5 != 4,:]
        y[2] =  F.interpolate(self.neck3(y[2][:, :, torch.arange(Wg) % 5 != 4,:].permute(0, 3, 1, 2)), scale_factor=4, mode='bilinear', align_corners=False)  ##[:, :, torch.arange(Wg) % 5 != 4,:]
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
    pad_w = (window_size +1 - W % (window_size + 1)) % (window_size + 1)  #+1 has beend added to the width dimension to compensate for the added channel dimension
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // (window_size + 1), window_size + 1, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size +1, C)
    return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
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
    B = windows.shape[0] // (Hp * Wp // window_size // (window_size+1))
    x = windows.view(B, Hp // window_size, Wp // (window_size+1), window_size, window_size + 1, -1)
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



