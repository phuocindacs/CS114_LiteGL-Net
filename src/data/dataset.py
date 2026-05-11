import os
import random
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class LOLDataset(Dataset):
    def __init__(self, root_dir, mode='train', patch_size=256):
        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        
        self.data_dir = os.path.join(root_dir, 'our485' if mode == 'train' else 'eval15')
        self.low_dir = os.path.join(self.data_dir, 'low')
        self.high_dir = os.path.join(self.data_dir, 'high')
        
        self.image_names = sorted([f for f in os.listdir(self.low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        low_img = Image.open(os.path.join(self.low_dir, img_name)).convert('RGB')
        high_img = Image.open(os.path.join(self.high_dir, img_name)).convert('RGB')
        
        if self.mode == 'train':
            w, h = low_img.size
            if w >= self.patch_size and h >= self.patch_size:
                i, j, th, tw = T.RandomCrop.get_params(low_img, output_size=(self.patch_size, self.patch_size))
                low_img = TF.crop(low_img, i, j, th, tw)
                high_img = TF.crop(high_img, i, j, th, tw)
                
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)

            if random.random() > 0.5:
                low_img = TF.vflip(low_img)
                high_img = TF.vflip(high_img)

            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                low_img = TF.rotate(low_img, angle)
                high_img = TF.rotate(high_img, angle)

        return TF.to_tensor(low_img), TF.to_tensor(high_img), img_name

class LOLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, patch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

    def setup(self, stage=None):
        self.train_dataset = LOLDataset(self.data_dir, mode='train', patch_size=self.patch_size)
        self.val_dataset = LOLDataset(self.data_dir, mode='eval', patch_size=self.patch_size)
        self.test_dataset = LOLDataset(self.data_dir, mode='eval', patch_size=self.patch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)
