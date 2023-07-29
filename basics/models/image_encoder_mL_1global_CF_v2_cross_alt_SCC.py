import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Tuple, Type
from functools import partial
from .SAM_commons import MLPBlock, LayerNorm2d
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
                torch.zeros(1, img_size // 4, img_size // 4, 48)   #*changed from patch_size to 4   img_size // 4 changed to 160
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
        self.chan_block2 = CAttentionBlock(
                embedding_dim = 48,
                num_heads = num_heads,
                shift_size= 1,
                
            )
        
        #TODO  down c4 and c2 blocks
        ''' c4 and c2 blocks further channel fusion

        self.pos_embedc4 = nn.Parameter(
            torch.zeros(1, img_size // 4, img_size // 4, 48)
        )

        self.c4_blocks  = nn.ModuleList()
        c4_depth = 3   #depth of c4 blocks
        for i in range(c4_depth):
            block = CAttentionBlock(
                embedding_dim = 48,
                num_heads = num_heads,
            )
            self.c4_blocks.append(block)

        self.c2_blocks = nn.ModuleList()
        c2_depth = 3
        for i in range (c2_depth):
            block = C2AttentionBlock(
                embedding_dim= 96,
                num_heads= num_heads
            )
        '''
        
        

        #swin blocks
        shift_size = [0, 2, 0, 2, 0, 2, 0, 2]
        self.stage1 = nn.ModuleList()
        for i in range(8):
            block = SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(128,128),
                num_heads=num_heads,
                window_size=8,
                shift_size=shift_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
                linear_mlp= shift_size[i] == 0
            )
            self.stage1.append(block)

        self.pmerging1 = PatchMerging((128, 128), embed_dim)

        self.stage2 = nn.ModuleList()
        for i in range(3):
            block = SwinTransformerBlock(
                dim=384,
                input_resolution=(64,64),
                num_heads=num_heads,
                window_size=8,
                shift_size=shift_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
            )
            self.stage2.append(block)

        self.pmerging2 = PatchMerging((64, 64), 384)

        self.stage3 = nn.ModuleList()
        for i in range(1):
            block = SwinTransformerBlock(
                dim=768,
                input_resolution=(32,32),
                num_heads=num_heads,
                window_size=32,
                shift_size=shift_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
            )
            self.stage3.append(block)


        '''
        previous blocks
        top_padding = [False, True]
        self.stage1 = nn.ModuleList()
        for i in range(2):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=3,#window_size[i] if i not in global_attn_indexes else 0,
                top_padding = False,#top_padding[i],
                input_size=(img_size // 4, img_size // 4),

            )
            self.stage1.append(block)
        self.pmerging1 = PatchMerging((128, 128), embed_dim)
        #stage2

        self.stage2 = nn.ModuleList()
        for i in range(2):
            block = Block(
                dim=384,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=3,#window_size[i] if i not in global_attn_indexes else 0,
                top_padding = False,#top_padding[i],
                input_size=(img_size // 8, img_size // 8),

            )
            self.stage2.append(block)
        self.pmerging2 = PatchMerging((64, 64), 384)
        #stage3
        self.stage3 = nn.ModuleList()
        for i in range(2):
            block = Block(
                dim=768,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=3,#window_size[i] if i not in global_attn_indexes else 0,
                top_padding = False,#top_padding[i],
                input_size=(img_size // 16, img_size // 16),

            )
            self.stage3.append(block)
        #TODO following layer is huge and not used
        ##self.fc_layer = MLPBlock(embedding_dim=192, mlp_dim=384, act=nn.GELU)
        #nn.Linear(192, 192) #removed (img_size //16) *
        
        '''
    
        #a second pathc embedding
      
        self.neck3 = nn.Conv2d(
                    768,
                    512,
                    kernel_size=1,
                    bias=False,
            )
           

        self.neck2 = nn.Conv2d(
                384,
                256,
                kernel_size=1,
                bias=False,
            )
        
        self.neck1 = nn.Conv2d(
            384,
            128,
            kernel_size=1,
            bias=False,
        )

        """
        self.neck1 = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                384,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(384),
            nn.Conv2d(
                384,
                384,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(384),
        )
        """
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
        
        r, g ,b, i = get_channels(x)
        r = self.channel_embed_r(r)#.unsqueeze(1)      #x1[:,0,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        g = self.channel_embed_g(g)#.unsqueeze(1)      #x1[:,1,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        b = self.channel_embed_b(b)#.unsqueeze(1)      #x1[:,2,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)
        i = self.channel_embed_i(i)#.unsqueeze(1)       #x1[:,3,:,:].unsqueeze(1)).view(bs, 1, 1, 1536)

        if self.pos_embed is not None:
            if r.shape[1] == self.pos_embed.shape[1]: #patches before
                r = r + self.pos_embed
                g = g + self.pos_embed
                b = b + self.pos_embed
                i = i + self.pos_embed

        r, g, b, i = self.chan_block(r, g, b, i)
        r, g, b, i = self.chan_block2(r, g, b, i)
        x = torch.cat((r, g, b, i), dim=-1)
        x = x.permute(0, 3, 1, 2)
        
        x = self.patch_embed(x)

        """following c4 and c2 blocks
        #adding position embeddings to each channel separately
        if self.pos_embedc4 is not None:
            if r.shape[1] == self.pos_embedc4.shape[1]: #patches before
                r = r + self.pos_embedc4
                g = g + self.pos_embedc4
                b = b + self.pos_embedc4
                i = i + self.pos_embedc4

        #C4 blocks preferabily with window size of 3
        for j in range(len(self.c4_blocks)):
            r, g, b, i = self.c4_blocks[j](r, g, b, i)
        
        c1 = torch.cat((r,g), dim=-1)
        c2 = torch.cat((b,i), dim = -1)
        
        #TODO add 1 or 2 fully connected for c1 and c2 to pass through for mixing up the r-g and b-i  optional

        #C2 attention blocks with window size of 7
        for i in range(len(self.c2_blocks)):
            c1, c2 = self.c2_blocks[i](r, g, b, i, 7)   #7 is the size of window used for window attention
        x = torch.cat((c1, c2), dim = -1)
        
        #TODO potential for 1 fullcy connected layer to mix up c1 and c2

        #breakpoint()
        """
        # x = self.chan_block(r, g, b, i)
        # x = x.permute(0, 3, 1, 2)
        # x = self.patch_embed(x)

        y = []
        # if self.pos_embed is not None:
        #     if x.shape[1] == self.pos_embed.shape[1]: #patches before
        #         x = x + self.pos_embed
        #x = x1 + patches
        bs, h, w, c = x.shape
        x = x.view(bs, h*w, c)
        z= []
        for i in range(len(self.stage1)):
            x = self.stage1[i](x)
            if i in (6, 7):
                x = x.view(bs, h, w, c)
                z.append(x)
                x = x.view(bs, h * w, c)
        
        y.append(torch.cat(z, dim=-1))
        #x = x.view(bs, h * w, c)
        x = self.pmerging1(x, (h, w))



        #stage2
        for i in range(len(self.stage2)):
            x = self.stage2[i](x)
        x = x.view(bs, h//2, w//2, -1)
        y.append(x)
        bs, h, w, c = x.shape
        x = x.view(bs, h * w, c)
        x = self.pmerging2(x, (h, w))


        #stage3
        for i in range(len(self.stage3)):
            x = self.stage3[i](x)
        x=x.view(bs, h//2, w//2, -1)
        y.append(x)

        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.neck(x.permute(0, 3, 1, 2))
        W = y[0].shape[2]
        Wg = x.shape[2]
        #y[0] = y[0] + x
        y[0] = self.neck1(y[0].permute(0, 3, 1, 2)) ##[:, :, torch.arange(W) % 5 != 4,:]
        y[1] = self.neck2(y[1].permute(0, 3, 1, 2)) #F.interpolate( , scale_factor=2, mode='bilinear', align_corners=False) ##[:, :, torch.arange(W) % 5 != 4,:]
        y[2] =  self.neck3(y[2].permute(0, 3, 1, 2))    #F.interpolate(, scale_factor=4, mode='bilinear', align_corners=False)  ##[:, :, torch.arange(Wg) % 5 != 4,:]
        #y[0] = y[0] + y[2]
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
        top_padding: bool = False,
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
        # self.lin1 = nn.Linear(dim, dim)
        # self.conv1 = nn.Conv2d(dim , dim, 2)
        # self.gelu = nn.GELU()
        # self.conv2 = nn.Conv2d(dim, dim, 2)
        # self.lin2 = nn.Linear(dim, dim)
        self.window_size = window_size
        self.top_padding = top_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]

            x, pad_hw = window_partition(x, self.window_size, self.top_padding)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W), self.top_padding)
            
        x = shortcut + x

        #nshortcut = x
        # x= self.lin1(self.norm2(x))
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, (0, 1, 0, 1))
        # x = self.conv1(x)
        # x = F.pad(x, (0, 1, 0, 1))
        # x = self.gelu(x)
        # x = self.conv2(x)
        # x = x.permute(0, 2, 3, 1)
        # x = self.lin2(x)
        # x = x + nshortcut
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
        shift_size =0,
        ) -> None:
        '''
        transformer block for calculating intra channel attention for 4 channels
        '''

        super().__init__()


        self.r2g_attn = CAttention(embedding_dim, num_heads, shift_size)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.rg2b_attn = CAttention(embedding_dim, num_heads, shift_size)
        self.norm2 = nn.LayerNorm(embedding_dim)


        self.rgb2ir_attn = CAttention(embedding_dim, num_heads, shift_size)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.ir2rgb_attn = CAttention(embedding_dim, num_heads, shift_size)
        self.norm4 = nn.LayerNorm(embedding_dim)
        
        self.window_size = 2
        self.input_resolution = (128, 128)
        self.shift_size = shift_size
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows, _ = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        # self.fc_layer = nn.Linear (192, out_dim)
        # #self.dropout = nn.Dropout(0.4)
        # self.mlp = MLPBlock(out_dim, 256, activation)
        # self.norm5 = nn.LayerNorm(out_dim)

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, ir: torch.Tensor, window_size:int = 2):
        b1, h, w, c = r.shape
        # r, r_hw =window_partition(r, window_size)
        # g, g_hw = window_partition(g, window_size)
        # b, b_hw = window_partition(b, window_size)
        # ir, ir_hw = window_partition(ir, window_size)
        # b2, h2, w2, c2 = r.shape

        if self.shift_size > 0:
            #shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_r = torch.roll(r, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_g = torch.roll(g, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_b = torch.roll(b, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_ir =torch.roll(ir, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            #partition of windows
            r_w, r_hw = window_partition(shifted_r, self.window_size)
            g_w, g_hw = window_partition(shifted_g, self.window_size)
            b_w, b_hw = window_partition(shifted_b, self.window_size)
            ir_w, ir_hw = window_partition(shifted_ir, self.window_size)
            b2, h2, w2, c2 = r_w.shape
            # partition windows
            #x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            
        else:
            #shifted_x = x
            # partition windows
            #x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            r_w, r_hw =window_partition(r, window_size)
            g_w, g_hw = window_partition(g, window_size)
            b_w, b_hw = window_partition(b, window_size)
            ir_w, ir_hw = window_partition(ir, window_size)
            b2, h2, w2, c2 = r_w.shape

        r_w = r_w.reshape(b2, h2 * w2, c2)
        g_w = g_w.reshape(b2, h2 * w2, c2)
        b_w = b_w.reshape(b2, h2 * w2, c2)
        ir_w = ir_w.reshape(b2, h2 * w2, c2)

        r_out = self.r2g_attn(q = r_w, k =g_w, v =g_w, dimensions =(h, w), mask =self.attn_mask)
        #x1 = r + r_out
        #x1 = self.norm1(x1)

        g_out = self.rg2b_attn(q = g_w, k =b_w, v =b_w, dimensions =(h, w), mask =self.attn_mask)
        #x2 = g + g_out
        #x2 = self.norm2(x2)


        b_out = self.rgb2ir_attn(q =b_w, k =ir_w, v =ir_w, dimensions =(h, w), mask =self.attn_mask)
        #x3 = b + b_out
        #x3 = self.norm3(x3)

        ir_out = self.ir2rgb_attn(q = ir_w, k =g_w, v =g_w, dimensions =(h, w), mask =self.attn_mask)
        #x4 = ir + ir_out
        #x4 = self.norm4(x4)

        r_out = r_out.view(b2, h2, w2, c2)
        g_out = g_out.view(b2, h2, w2, c2)
        b_out = b_out.view(b2, h2, w2, c2)
        ir_out = ir_out.view(b2, h2, w2, c2)

        if self.shift_size > 0:
            shifted_r = window_unpartition(r_out, self.window_size, r_hw, (h, w))  # B H' W' C
            shifted_g =  window_unpartition(g_out, self.window_size, g_hw, (h, w))
            shifted_b =  window_unpartition(b_out, self.window_size, b_hw, (h, w))
            shifted_ir =  window_unpartition(ir_out, self.window_size, ir_hw, (h, w))
            r_out = torch.roll(shifted_r, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            g_out = torch.roll(shifted_g, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            b_out = torch.roll(shifted_b, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            ir_out = torch.roll(shifted_ir, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            #shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            r_out = window_unpartition(r_out, self.window_size, r_hw, (h, w))  # B H' W' C
            g_out =  window_unpartition(g_out, self.window_size, g_hw, (h, w))
            b_out =  window_unpartition(b_out, self.window_size, b_hw, (h, w))
            ir_out =  window_unpartition(ir_out, self.window_size, ir_hw, (h, w))
            #x = shifted_x

        x1 = self.norm1(r + r_out)
        x2 = self.norm2(g + g_out)
        x3 = self.norm3(b + b_out)
        x4 = self.norm4(ir + ir_out)
        

        #x = torch.cat((x1, x2, x3, x4), dim=-1)
        # x = self.fc_layer(x)
        # #x = self.dropout(x)
        # x = self.mlp(x)
        # x = self.norm5(x)
        return x1, x2, x3, x4




class C2AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        out_dim: int = 192,
        activation: Type[nn.Module] = nn.ReLU,
        skip_pe: bool = True,
        ) -> None:
        '''
        transformer block for calculating intra channel attention for 2 channels
        '''

        super().__init__()


        self.c12c2_attn = CAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.c22c1_attn = CAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)


        # self.rgb2ir_attn = CAttention(embedding_dim, num_heads)
        # self.norm3 = nn.LayerNorm(embedding_dim)

        # self.ir2rgb_attn = CAttention(embedding_dim, num_heads)
        # self.norm4 = nn.LayerNorm(embedding_dim)
        
       

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, window_size:int):
        b1, h, w, c = r.shape
        r, r_hw =window_partition(c1, window_size)
        g, g_hw = window_partition(c2, window_size)
        
        b2, h2, w2, c2 = r.shape
        r = r.reshape(b2, h2 * w2, c2)
        g = g.reshape(b2, h2 * w2, c2)


        attn_out = self.r2g_attn(q = r, k =g, v =g)
        x1 = r + attn_out
        x1 = self.norm1(x1)

        attn_out = self.rg2b_attn(q = g, k =r, v =r)
        x2 = g + attn_out
        x2 = self.norm2(x2)


        x1 = x1.view(b2, h2, w2, c2)
        x2 = x2.view(b2, h2, w2, c2)

        x1 = window_unpartition(x1, window_size, r_hw, (h, w))
        x2 = window_unpartition(x2, window_size, g_hw, (h, w))

        x = torch.cat((x1, x2), dim=-1)
        # x = self.fc_layer(x)
        # #x = self.dropout(x)
        # x = self.mlp(x)
        # x = self.norm5(x)
        return x1, x2




class CAttention(nn.Module):
    """attention layer allowing cross attention used for channels"""

    def __init__(self, 
                embedding_dim:int,
                num_heads: int= 8,
                shift_size = 0
                ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp = Mlp(embedding_dim, embedding_dim * 4, embedding_dim, linear_mlp=True)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dimensions, mask = None) -> torch.Tensor:
        '''
        -dimensions: tuple(int, int) the original dimension of feature map before window partitioning in height and width
        '''
        B_, N, C = q.shape
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        #Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            #attn = self.softmax(attn)
        
            #attn = self.softmax(attn)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        #output
        out = attn @ v
        out = self._recombine_heads(out)
        out = out + self.mlp(out, dimensions[0], dimensions[1])  #128 is hard coded height and width
        return out


def window_partition(x: torch.Tensor, window_size: int, top_padding: bool = False) -> Tuple[torch.Tensor, Tuple[int, int]]:
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
        if top_padding:
            x = F.pad(x, (0, 0, pad_w, 0, pad_h, 0))
        else:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int], top_padding: bool = False
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
        if top_padding:
            x = x[:, -H:, -W:, :].contiguous()
        else:
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


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim * 2, bias=False)
        self.norm = norm_layer(dim * 2)

    def forward(self, x, input_resolution):
        """
        x: B, H*W, C
        """
        H, W = input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, linear_mlp=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear = linear_mlp
        self.bs = in_features
        if self.linear:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        else:
            self.fc1 = nn.Linear(in_features, in_features)
            self.act = act_layer()
            self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1, groups=in_features)
            #self.conv2 = nn.Conv2d(in_features, in_features, 2)
            self.fc2 = nn.Linear(in_features, out_features)
            self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        if self.linear:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            x = self.fc1(x)
            bs = x.shape[0]
            x = x.permute(0, 2, 1).contiguous()
            x = x.view(bs, -1, H, W)
            #x = F.pad(x, (0, 1, 0, 1))
            x = self.conv1(x)
            #x = F.pad(x, (0, 1, 0, 1))
            #x = self.conv2(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(bs, H * W, -1)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)


        return x


##Swin codes

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, linear_mlp = True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, linear_mlp = linear_mlp )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows, phw = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows, phw = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_unpartition(attn_windows, self.window_size, phw, (H, W))  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_unpartition(attn_windows, self.window_size,phw, (H, W))  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

