import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class CustomData(Dataset):

    def __init__(self, split,im_path, img_size, crop_size, im_ext='png'):

        self.split = split
        self.im_ext = im_ext
        self.images = self.load_images(im_path)
        self.crop_size = crop_size
        self.img_size = img_size

    
    def load_images(self, im_path):

        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        # labels = []
        for d_name in tqdm(os.listdir(im_path)):
            d_name = os.path.join(im_path,d_name)
            ims.append(d_name)
            # labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im = torchvision.transforms.CenterCrop(self.crop_size)(im)
        im = torchvision.transforms.Resize(self.img_size)(im)

        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
