import torch
from models.transformer_discriminator import Discriminator
from models.esrt import ESRT
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = torch.rand(4,1,200,200).to(device)

discriminator = Discriminator(img_size=200,patch_size=10,in_chans=1, num_classes=1, embed_dim=500, num_heads=4,mlp_ratio=4.,
qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.,depth=4,act_layer='gelu',diff_aug='None',norm_layer='ln').to(device=device)
attrs = vars(discriminator)
# {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
# now dump this in some way or another
print(', '.join("%s: %s" % item for item in attrs.items()))


# print(discriminator)
output_tensor =  discriminator(input_tensor)
print(output_tensor.shape)
print(output_tensor)


# generator = ESRT().to(device=device)
# # print(generator)

# output_tensor = generator(input_tensor)
# print(output_tensor.shape)
# quit();


## 3d rrdbnet
# input_tensor = torch.rand(2,1,100,100,100).to(device)
# rrdbnet =  rrdbnet_x2().to(device)
# output_tensor = rrdbnet(input_tensor)
# print(output_tensor.shape)


# test_tensor = torch.rand(1,3,700,700).to(device).flatten(2)
# print(test_tensor.shape)

# B = 32
# cls_token = nn.Parameter(torch.zeros(1, 1, 500))
# cls_token = cls_token.expand(B, -1, -1)
# print(cls_token.shape)

# test_tensor = torch.rand(1,101,500)[:,0]
# print(test_tensor.shape)