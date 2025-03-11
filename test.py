from datasets.larynx_data import Larynx_Data
from monai.transforms import IdentityD, Compose, ResizeD, ScaleIntensityD, LoadImageD, EnsureChannelFirstd, RandFlipD, RandRotate
import numpy as np

print (5%4)
augmentations = [
        ScaleIntensityD(keys=["image"]),
        RandFlipD(keys=["image"], spatial_axis=0),
        RandRotate(range_x=np.pi/6)
]

dataset_train = Larynx_Data(root='DATA', mode="train", spatial_size=128, augmentations=augmentations)
print(dataset_train.__getitem__(1)["image"].shape)
length = dataset_train.__len__()
print(length)