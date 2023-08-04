import torch
from models.transformer_discriminator import Discriminator, RankDiscriminator
from models.esrt import ESRT
import torch.nn as nn
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import os

import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# classification_model = RankDiscriminator(patch_size=10, in_chans=1, num_classes=5,embed_dim=500,
#                 num_heads=4, mlp_ratio=4.,qkv_bias=False,qk_scale=None, drop_rate=0.,norm_layer='ln',
#                 depth=4,act_layer='gelu', diff_aug='None',apply_sigmoid=False)

# classification_model.to(device)

# # classification_model.eval()

# input = torch.rand(1,1,200,200).to(device=device)
# output = classification_model(input)
# # target = torch.tensor([5]).to(device)
# target = torch.empty(1, dtype=torch.long).random_(5).to(device=device)

# print("shape of output",output.shape)
# print("output of model",output)
# print("taget", target)

# loss = nn.CrossEntropyLoss()
# loss_value = loss(output, target)
# print("loss value is", loss_value.item())

# optimizer = optim.Adam(classification_model.parameters(), lr=0.001)
# num_epochs = 50


# for epoch in range(num_epochs):
#     classification_model.train()
#     running_loss = 0.0

#     for i in range(50):
#         inputs = torch.rand(1,1,200,200).to(device=device)
#         labels = torch.empty(1, dtype=torch.long).random_(5).to(device=device)
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = classification_model(inputs)
#         loss_val = loss(outputs, labels)
#         loss_val.backward()
#         optimizer.step()

#         running_loss += loss_val.item()

#     epoch_loss = running_loss / 50
#     if epoch % 10 == 0:
#         path = os.path.join('demo_weights/', 'epoch_{}_f_{}.pth'.format(epoch,2))
#         classification_model.save(model_weights=classification_model.state_dict(),epoch=epoch,save_optimizer=False,
#         path= path )

#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# print("Training finished!")



def load_state_dict_func(path):

    # state_dict = torch.load(path)
    state_dict=path
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    model = RankDiscriminator(
           patch_size = checkpoint['patch_size'], 
           in_chans = checkpoint['in_chans'], 
           num_classes = checkpoint['num_classes'],
           embed_dim = checkpoint['embed_dim'],
           num_heads = checkpoint['num_heads'], 
           mlp_ratio = checkpoint['mlp_ratio'],
           qkv_bias = checkpoint['qkv_bias'],
           qk_scale = checkpoint['qk_scale'], 
           drop_rate = checkpoint['drop_rate'],
           norm_layer = checkpoint['norm_layer'],
           depth = checkpoint['depth'],
           act_layer = checkpoint['act_layer'],
           diff_aug = checkpoint['diff_aug'],
           apply_sigmoid = checkpoint['apply_sigmoid']
            ).to(device = device)

    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict,strict=False)
    return model


loaded_model1 = load_model(checkpoint_path='demo_weights/epoch_0_f_2.pth', device=device)
input = torch.rand(1,1,200,200).to(device=device)
output1 = loaded_model1(input)
loaded_model2 = load_model(checkpoint_path='demo_weights/epoch_40_f_2.pth', device=device)
output2 = loaded_model2(input)

print(output1)
print(output2)