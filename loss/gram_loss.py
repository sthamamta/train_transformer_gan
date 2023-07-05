
import torch
from torch import nn 

# measures the mse or l1 loss between the gram matrix of output and label images
class GramLoss(nn.Module):
    def __init__(self, diff='mse'):
        super().__init__()
        if diff == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()

    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram 

    def __call__(self, images, labels):
        image_gram = self.gram_matrix(images)
        label_gram = self.gram_matrix(labels)
        loss = self.criterion(label_gram,image_gram)
        return loss
