"""
Debugging script. Meant to be run interactively (f.e. in a ipython session)
to inspect the output of a model and its intermediate activations.
"""
from models import load_model
from PIL import Image
from utils import pil_to_tensor, tensor_to_pil
import os
import torch

# Variables
# --------------------------------------------------

checkpoint = "checkpoints/fp_gdn/checkpoint-model=fp_gdn-lambda=0.1-best.pth.tar"
image = os.path.expanduser("~/Downloads/chess-screenshot-box.png")

# --------------------------------------------------

model = load_model(checkpoint)

# this hook can be used to inspect the intermediate values of a model
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().squeeze()
    return hook


# registering the hook with the model
for layer in [model.synthesis_transform, model.analysis_transform]:
    for name, module in layer.named_children():
        module.register_forward_hook(get_activation(name))

# tracing an image through the model for intermediate values
with Image.open(image) as im:
    x = pil_to_tensor(im)
    compressed, y_quant = model.compress_constriction(x)
    x_hat = model.decompress_constriction(compressed, y_quant.shape)

    print(f"\nImage size: {compressed.size*4} Byte")
    im_hat = tensor_to_pil(x_hat)
    # im_hat.show()

activation["x"] = x
activation["y_quant"] = y_quant
activation["x_hat"] = x_hat

# might have to be commented out if the model has a different build
assert torch.all(torch.nn.functional.conv2d(
    x, model.analysis_transform[0].weight, stride=2, padding=2).squeeze() == activation["conv0"])

print(f"\nCaptured output of layers: {list(activation.keys())}")
print("\nAccess via activation['layer_name']\n")
