import torch
import torch.nn as nn
import math
import numpy as np


from models.discriminator_helper import DropPath, to_2tuple, trunc_normal_
from models.diff_aug import DiffAugment

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])

class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu
        
    def forward(self, x):
        return self.act_layer(x)
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale= None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5  
        self.scale = head_dim ** -0.5
        # print("***********************************************")
        # print("scale in attention nw discriminator", self.scale)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x):
        # print("INput to attention module", x.shape)
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # print("Before Error", self.scale)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("Output to attention module",x.shape)
        return x

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)
        
    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0,2,1)).permute(0,2,1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)




def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class DisBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x*self.gain + self.drop_path(self.attn(self.norm1(x)))*self.gain
        x = x*self.gain + self.drop_path(self.mlp(self.norm2(x)))*self.gain
        return x


class Discriminator(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,img_size=100,in_chans=1, num_classes=1, embed_dim= 900,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,norm_layer='ln', depth=2, act_layer = 'gelu', patch_size = 10, diff_aug='None', apply_sigmoid=False):
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale= qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        self.num_features =  self.embed_dim = embed_dim
        
        self.depth = depth
        self.act_layer = act_layer
        self.patch_size = patch_size #patch size to treat each patch as a token
        self.diff_aug = diff_aug


        # if hybrid_backbone is not None:
        #     self.patch_embed = HybridEmbed(
        #         hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        # else:
        self.patch_embed = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        num_patches = (img_size // self.patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DisBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for i in range(depth)])
        
        self.norm = CustomNorm(self.norm_layer, self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.apply_sigmoid = apply_sigmoid
        self.sig = torch.nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Conv2d) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        if "None" not in self.diff_aug:
            x = DiffAugment(x, self.diff_aug, True)
        B = x.shape[0]
        # print("shape of x before", x.shape)
        # print("shape of patch embedding", self.patch_embed(x).shape)
        x = self.patch_embed(x).flatten(2).permute(0,2,1) # flatten featuremap for each channel and aranage into (batch, flatten_featuremap, channel)

        # print("size of x after patch_enbed and flatten", x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks # create cls token for each image in a batch
        # print("size of cls token", cls_tokens.shape)

        x = torch.cat((cls_tokens, x), dim=1)
        # print("size of x after concatenating cls token", x.shape)

        # print("shape of position embedding", self.pos_embed.shape)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            # print("input of each block", x.shape)
            x = blk(x)
            # print("output of each block", x.shape)

        x = self.norm(x)
        # print("shape of x at output", x.shape)
        return x[:,0] #select all elements along dim 0

    def forward(self, x):
        x = self.forward_features(x)
        # print("before passing through head", x.shape)
        x = self.head(x)
        if self.apply_sigmoid:
            x = self.sig(x)
        return x




#*******************************************    3D Discriminator    *********************************************************************

class Discriminator3D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,img_size=700, patch_size= (64,64), in_chans=1, num_classes=1, embed_dim=None, depth=2,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer='ln', d_depth=6, act_layer = 'gelu', df_dim=500, p_size = 64, diff_aug='color'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = df_dim  
        
        depth = d_depth
        # self.args = args
        patch_size = p_size
        norm_layer = norm_layer
        act_layer = act_layer
        mlp_ratio = mlp_ratio
        self.diff_aug = diff_aug


        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (img_size // patch_size)**3

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DisBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for i in range(depth)])
        
        self.norm = CustomNorm(norm_layer, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        if "None" not in self.diff_aug:
            x = DiffAugment(x, self.diff_aug, True)
        B = x.shape[0]
        # print("shape of x before", x.shape)
        # print("shape of patch embedding", self.patch_embed(x).shape)
        x = self.patch_embed(x).flatten(2).permute(0,2,1) # flatten featuremap for each channel and aranage into (batch, flatten_featuremap, channel)

        # print("size of x after patch_enbed and flatten", x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks # create cls token for each image in a batch
        # print("size of cls token", cls_tokens.shape)

        x = torch.cat((cls_tokens, x), dim=1)
        # print("size of x after concatenating cls token", x.shape)

        # print("shape of position embedding", self.pos_embed.shape)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            # print("input of each block", x.shape)
            x = blk(x)
            # print("output of each block", x.shape)

        x = self.norm(x)
        # print("shape of x at output", x.shape)
        return x[:,0] #select all elements along dim 0

    def forward(self, x):
        x = self.forward_features(x)
        # print("before passing through head", x.shape)
        x = self.head(x)
        if self.apply_sigmoid:
            x = self.sig(x)
        return x
