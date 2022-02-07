from models import FactorizedPrior
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from typing import Dict


def tensor_to_pil(x: torch.Tensor) -> Image:
    """Convert a tensor to a PIL image."""
    return ToPILImage()(x.squeeze())


def pil_to_tensor(im: Image) -> torch.Tensor:
    """Convert a PIL image to a tensor."""
    tensor = ToTensor()(im)
    if tensor.shape[0] == 3:
        return tensor[None, :, :, :]
    else:
        return tensor[None, :3, :, :]


im_path = "~/projects/CompressAI/link_180x210.png"
model = FactorizedPrior(128, 192)


def clean_checkpoint_data_parallel(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    # If the model was trained with DataParallel, we will have to remove
    # the .module prefix from the keys. 
    # See:
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/8
    new_dict = {}
    for key in checkpoint["state_dict"]:
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_dict[new_key] = checkpoint["state_dict"][key]
    return new_dict


checkpoint = torch.load("checkpoint_best_loss.pth.tar",
                        map_location=torch.device("cpu"))

new_dict = clean_checkpoint_data_parallel(checkpoint)
model.load_state_dict(new_dict)
model.update()

with Image.open("/Users/almico/Downloads/link.png") as im:
    x = pil_to_tensor(im)
    s = model.compress(x)
    x_hat = model.decompress(s["strings"], s["shape"])["x_hat"]
    im_hat = tensor_to_pil(x_hat)
    im_hat.show()
