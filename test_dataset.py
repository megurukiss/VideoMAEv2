from dataset import spatial_sampling,random_short_side_scale_jitter
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

# generate random image shape (3,2,256,349)
# image_shape = (3,2,256, 349)
# buffer=torch.rand(image_shape)
# # generate random spatial size
# scl, asp = (
#             [0.08, 1.0],
#             [0.75, 1.3333],
#         )

# frames, _ = random_short_side_scale_jitter(
#                 images=buffer,
#                 min_size=224,
#                 max_size=224,
#                 inverse_uniform_sampling=False,
#             )

# img=spatial_sampling(buffer,
#             spatial_idx=-1,
#             min_scale=224,
#             max_scale=224,
#             crop_size=224,
#             random_horizontal_flip=False,
#             inverse_uniform_sampling=False,
#             aspect_ratio=asp,
#             scale=scl,
#             motion_shift=False)
# print(frames.shape)

import torch
import torchvision.transforms as transforms

# Example tensor
C, T, H, W = 3, 10, 240, 320  # Channels, Time, Initial Height, Initial Width
buffer = torch.rand(C, T, H, W)  # Random data for demonstration

# Define the resize transform
resize_transform = transforms.Resize((224, 224))

def resize_and_pad(tensor, size=224):
    # Get the current dimensions from the tensor
    C, T, H, W = tensor.shape
    
    # Determine the scaling factor and the new dimensions
    scale = size / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    
    # Resize the image
    resized = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Calculate padding
    pad_h = (size - new_h + 1) // 2 if new_h < size else 0
    pad_w = (size - new_w + 1) // 2 if new_w < size else 0
    
    # Apply padding
    padded = F.pad(resized, [pad_w, pad_h, size - new_w - pad_w, size - new_h - pad_h], mode='constant', value=0)
    
    return padded

# Resize each frame in the time dimension
resized_padded_frames = [resize_and_pad(buffer[:, i].unsqueeze(0)) for i in range(buffer.shape[1])]
resized_padded_buffer = torch.cat(resized_padded_frames, dim=0)
print(resized_padded_buffer.shape)

import matplotlib.pyplot as plt
# show the first image 


