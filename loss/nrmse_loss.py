import torch.nn as  nn
import torch

class NRMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.eps = eps
        
    def forward(self,y,yhat):
        numerator = torch.sqrt(self.mse(yhat,y) + self.eps)
        zeros = torch.zeros(y.shape).to(y.get_device())
        denominator = torch.sqrt(self.mse(y,zeros)+self.eps)
        loss = numerator/denominator
        return loss