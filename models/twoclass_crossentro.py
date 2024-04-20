import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoClassLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
            softmax [[0.8,0.2]],
        """
        
        super(TwoClassLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
    
    def forward(self, x, target,threshould=0.7):
        logprobs = F.softmax(x, dim=-1)
        # get the max value of the logprobs, check if max exceeds the threshould 
        
        # 
        
            
        
            
        