from torchvision.datasets import ImageNet
import torch

imagenet_data = ImageNet(root="~/datasets/imagenette")
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)