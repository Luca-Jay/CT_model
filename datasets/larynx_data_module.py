from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from datasets.Larynx_Data import Larynx_Data


class Larynx_DataModule(pl.LightningDataModule):
    # This dataloader doesn't work for testing because we have a complicated setup for testing
    def __init__(self, data_dir: str = "./", batch_size: int = 2, spatial_size: int = 128, augmentations: list = None, hu_values: list = None):
        super().__init__()
        # all transforms are defined in the dataset class
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.spatial_size = spatial_size
        self.augmentations = augmentations
        self.hu_values = hu_values

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.dataset_train = Larynx_Data(root=self.data_dir, mode="train", spatial_size=self.spatial_size, augmentations=self.augmentations, hu_values=self.hu_values)
            self.dataset_val = Larynx_Data(root=self.data_dir, mode="val", spatial_size=self.spatial_size, augmentations=self.augmentations, hu_values=self.hu_values)
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=8, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=8, persistent_workers=True, prefetch_factor=2)