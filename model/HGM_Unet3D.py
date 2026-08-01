import torch
import torch.nn as nn
bn_track = True


class UpConv3D(nn.Module):
    def __init__(self, in_c, out_c=None):
        super().__init__()
        out_c = out_c or in_c
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(self.up(x))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention3D(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention3D, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop_ratio=0., attn_drop_ratio=0.):
        super(Block3D, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EGEM(nn.Module):


    def __init__(self, input_dim, num_heads=2, mlp_ratio=2.0, depth=1, drop_ratio=0.1):
        super(EGEM, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.depth = depth
        self.blocks = nn.Sequential(*[
            Block3D(dim=input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(input_dim)
        # self.grid_size = (img_size // patch_size) ** 3
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size, input_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        B, C, D, H, W = x.shape


        # [B, C, D, H, W] -> [B, D*H*W, C]
        x = x.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # x = x + self.pos_embed.expand(B, x.size(1), -1)  #

        x = self.blocks(x)
        x = self.norm(x)
        # [B, D*H*W, C] -> [B, C, D, H, W]
        x = x.transpose(1, 2).view(B, C, D, H, W)

        return x




class CHRE3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(CHRE3D, self).__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.residual_conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class MGFIM(nn.Module):

    def __init__(self, enc_channels, dec_channels, F_int):
        super(MGFIM, self).__init__()

        self.W_enc = nn.Sequential(
            nn.Conv3d(enc_channels, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_dec = nn.Sequential(
            nn.Conv3d(dec_channels, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)


        self.reverse_psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.adjust = nn.Conv3d(F_int, enc_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, enc_feature, dec_feature):

        enc_feature_plus = self.W_enc(enc_feature)
        dec_feature_plus = self.W_dec(dec_feature)

        fused = self.relu(enc_feature_plus + dec_feature_plus)

        forward_attn = torch.sigmoid(self.psi(fused))
        enc_attended = enc_feature_plus * forward_attn

        reverse_attn = torch.sigmoid(self.reverse_psi(fused))
        dec_attended = dec_feature_plus * reverse_attn

        enhanced_feature = self.adjust(enc_attended + dec_attended)
        return enhanced_feature


class EViT_ResUNet3D(nn.Module):
    def __init__(self, input_ch=1, output_ch=1, init_feats=32):
        super(EViT_ResUNet3D, self).__init__()

        self.encoder_block_1 = CHRE3D(input_ch, init_feats)
        self.encoder_block_2 = CHRE3D(init_feats, init_feats * 2)
        self.encoder_block_3 = CHRE3D(init_feats * 2, init_feats * 4)
        self.encoder_block_4 = CHRE3D(init_feats * 4, init_feats * 8)
        self.encoder_block_5 = CHRE3D(init_feats * 8, init_feats * 16)

        self.max_pool_1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.max_pool_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.max_pool_3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.max_pool_4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.svit_layer = EGEM(
            input_dim=init_feats * 16, 
            num_heads=2,
            mlp_ratio=2.0, 
            depth=1, 
            drop_ratio=0.1
        )

        self.gated_attention_layer4 = MGFIM(
            enc_channels = init_feats * 8,
            dec_channels = init_feats * 16,
            F_int = init_feats * 4 
        )
        self.gated_attention_layer3 = MGFIM(
            enc_channels = init_feats * 4, 
            dec_channels = init_feats * 8, 
            F_int=init_feats * 2 
        )
        self.gated_attention_layer2 = MGFIM(
            enc_channels = init_feats * 2,
            dec_channels = init_feats * 4,
            F_int=init_feats
        )
        self.gated_attention_layer1 = MGFIM(
            enc_channels = init_feats,
            dec_channels = init_feats * 2, 
            F_int=init_feats // 2
        )

        self.upsample_1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.upsample_3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.upsample_4 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        # self.upsample_4 = UpConv3D(init_feats * 16)  # 256 → 256,  4³→8³
        # self.upsample_3 = UpConv3D(init_feats * 8)  # 128 → 128,  8³→16³
        # self.upsample_2 = UpConv3D(init_feats * 4)  # 64  → 64,   16³→32³
        # self.upsample_1 = UpConv3D(init_feats * 2)  # 32  → 32,   32³→64³

        self.de_conv_block4 = ConvBlock(init_feats * 16 + init_feats * 8, init_feats * 8)
        self.de_conv_block3 = ConvBlock(init_feats * 8 + init_feats * 4, init_feats * 4)
        self.de_conv_block2 = ConvBlock(init_feats * 4 + init_feats * 2, init_feats * 2)
        self.de_conv_block1 = ConvBlock(init_feats * 2 + init_feats, init_feats)

        self.final_conv = nn.Conv3d(init_feats, output_ch, kernel_size=1, stride=1)  # 修正：应该是init_feats而不是16

    def forward(self, x):
        # ---------- encoder ----------
        enc1 = self.encoder_block_1(x)  # 64³,16
        enc2 = self.encoder_block_2(self.max_pool_1(enc1))  # 32³,32
        enc3 = self.encoder_block_3(self.max_pool_2(enc2))  # 16³,64
        enc4 = self.encoder_block_4(self.max_pool_3(enc3))  # 8³,128
        enc5 = self.encoder_block_5(self.max_pool_4(enc4))  # 4³,256

        enc5_vit = self.svit_layer(enc5)  # 4³,256

        # ---------- decoder ----------
        up4 = self.upsample_4(enc5_vit)  # 8³, init_feats*16
        # att4 = self.gated_attention_layer4(enc4,up4)
        dec4 = self.de_conv_block4(torch.cat([enc4, up4], dim=1))

        up3 = self.upsample_3(dec4)  # 16³, init_feats*8
        att3 = self.gated_attention_layer3(enc3,up3)
        dec3 = self.de_conv_block3(torch.cat([att3, up3], dim=1))

        up2 = self.upsample_2(dec3)
        # att2 = self.gated_attention_layer2(enc2,up2)
        dec2 = self.de_conv_block2(torch.cat([enc2, up2], dim=1))


        up1 = self.upsample_1(dec2)  # 64³, init_feats*2
        # att1 = self.gated_attention_layer1(enc1,up1)
        dec1 = self.de_conv_block1(torch.cat([enc1, up1], dim=1))

        return self.final_conv(dec1)


# if __name__ == "__main__":
#     model = EViT_ResUNet3D(init_feats=16)
#     summary(model, input_size=(1, 1, 64, 64, 64), device="cpu")
