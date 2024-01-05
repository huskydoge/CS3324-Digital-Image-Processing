from custom_dataset import SaliencyDataSet
import PIL
import numpy as np
from PIL import Image
dataset = SaliencyDataSet()
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

type = "pure"

def transform_map(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize(0, 1),
    ])

def save_map_to_png(map, path):
    # Convert the tensor to a NumPy array if it's not already
    if not isinstance(map, np.ndarray):
        map = map.detach().cpu().numpy()

    map = np.squeeze(map)

    # Normalize the array to be in the range [0, 255] if it's not already
    # This step is optional and depends on the data range in your tensor
    map = (255 * (map - np.min(map)) / np.ptp(map)).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(map)

    # Save the image
    image.save(path, format='PNG')

train_loader, test_loader = dataset.get_loader(type)
from torchvision import transforms
import torch

x = torch.Tensor([1,2,3])
print(x.sort())

for idx, (img_paths, map_paths, fix_map_paths) in enumerate(train_loader):
    print("info {}".format(idx))
    img_path = img_paths[0]
    map_path = map_paths[0]
    fix_map_path = fix_map_paths[0]
    img = PIL.Image.open(img_path)
    map = PIL.Image.open(map_path)
    map_tensor = transforms.ToTensor()(map)
    save_map_to_png(map_tensor, 'before.png')

    map_ = transform_map(224)(map)
    mapn = map_.view(-1)
    print(set(mapn))
    print(mapn.sort())
    exit(0)
    print(map_.shape)
    save_map_to_png(map_, 'after.png')
    exit(0)


