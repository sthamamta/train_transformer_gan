
import torch
from torch import nn 
from models.transformer_discriminator import RankDiscriminator
from collections import OrderedDict

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




# measures the mse or l1 loss between the gram matrix of output and label images
class RankLoss(nn.Module):
    def __init__(self, checkpoint_path='outputs/transformer_classifier/transformer_classifier_ranking_ph123hr/checkpoints/patch/patch-200/epoch_3500_f_2.pth', device = 'cuda'):
        super().__init__()
        
        self.ranker = load_model(checkpoint_path=checkpoint_path, device=device)

        for param in self.ranker.parameters():
            param.requires_grad = True

        self.label = torch.tensor(4).to(device)
        self.device = device
        self.cross_entrophy_loss = nn.CrossEntropyLoss()

    def __call__(self, images):
        prediction= self.ranker(images)
        batch_size = prediction.shape[0]
        expanded_labels = self.label.expand(batch_size)
        loss = self.cross_entrophy_loss(prediction, expanded_labels)
        return loss
