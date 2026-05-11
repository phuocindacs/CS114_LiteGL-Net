import os
import random
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class LOLDataset(Dataset):
    def __init__(self, root_dir, mode='train', dataset_type='real', patch_size=256):
        """
        root_dir: Đường dẫn đến thư mục chứa 'Real_captured' và 'Synthetic' (thường là thư mục 'LOL-v2')
        mode: 'train' hoặc 'test'
        dataset_type: 'real' hoặc 'synthetic'
        """
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.dataset_type = dataset_type.lower()
        self.patch_size = patch_size

        # Ánh xạ tên thư mục cha theo loại dataset
        type_folder = 'Real_captured' if self.dataset_type == 'real' else 'Synthetic'

        # Ánh xạ tên thư mục theo mode (chú ý viết hoa chữ cái đầu theo đúng ảnh)
        mode_folder = 'Train' if self.mode == 'train' else 'Test'

        # Xây dựng đường dẫn (VD: root_dir/Real_captured/Train)
        self.data_dir = os.path.join(self.root_dir, type_folder, mode_folder)

        # Tên thư mục chứa ảnh tối và sáng ở LOL-v2 là 'Low' và 'Normal'
        self.low_dir = os.path.join(self.data_dir, 'Low')
        self.high_dir = os.path.join(self.data_dir, 'Normal')

        self.image_names = sorted([f for f in os.listdir(self.low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # 1. Tạo tên file dự kiến cho ảnh High/Normal
        # (Ví dụ: 'low00690.png' -> 'normal00690.png')
        high_img_name = img_name.replace('low', 'normal')

        # 2. Xây dựng đường dẫn
        high_img_path = os.path.join(self.high_dir, high_img_name)

        # 3. Fallback: Đề phòng trường hợp dataset của bạn không đổi tên
        # (cả 2 bên đều tên giống nhau), ta check nếu file 'normal...' không tồn tại thì dùng tên gốc
        if not os.path.exists(high_img_path):
            high_img_path = os.path.join(self.high_dir, img_name)

        # 4. Load ảnh
        low_img = Image.open(os.path.join(self.low_dir, img_name)).convert('RGB')
        high_img = Image.open(high_img_path).convert('RGB')

        # Augmentation chuẩn theo bài báo (Crop 256, Flip, Rotate)
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
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, patch_size: int, dataset_type: str = 'real'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.dataset_type = dataset_type

    def setup(self, stage=None):
        self.train_dataset = LOLDataset(self.data_dir, mode='train', dataset_type=self.dataset_type, patch_size=self.patch_size)
        self.val_dataset = LOLDataset(self.data_dir, mode='test', dataset_type=self.dataset_type, patch_size=self.patch_size)
        self.test_dataset = LOLDataset(self.data_dir, mode='test', dataset_type=self.dataset_type, patch_size=self.patch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
