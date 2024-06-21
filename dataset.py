import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val", "test"]
        data_root = "violence_224/"
        self.data = [os.path.join(data_root, split, i) for i in os.listdir(os.path.join(data_root, split))]
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path)
        y = int(img_path.split("\\")[-1][0])
        x = self.transforms(x)
        return x, y

    def convert_to_tensor(self):
        dataset_tensors = []
        for img_path in self.data:
            img = Image.open(img_path)
            img_tensor = self.transforms(img)
            dataset_tensors.append(img_tensor)
        return torch.stack(dataset_tensors, dim=0)
class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        # 分割数据集、应用变换等
        # 创建 training, validation数据集
        self.train_dataset = CustomDataset("train")

        self.val_dataset = CustomDataset("val")

        self.test_dataset = CustomDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)