import os
from pathlib import Path
from utils.utils import window
from torch.utils.data import Dataset
from monai.transforms import IdentityD, Compose, ResizeD, ScaleIntensityD, LoadImageD, EnsureChannelFirstd, RandFlipD, RandRotate
import numpy as np
import nibabel as nib

import time

def log_timing(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


class Larynx_Data(Dataset):

    def __init__(self, root, mode="train", augmentations=None, spatial_size=128):
        # get the correct root paths
        self.mode = mode
        if mode == "train":
            folder = "TRAIN"
        elif mode == "val":
            folder = "VAL"
        elif mode == "test-normal":
            folder = os.path.join("TEST", "NORMAL")
        elif mode == "test-strangulation":
            folder = os.path.join("TEST", "STRANGULATION")
        elif mode == "test-hemorrhage":
            folder = os.path.join("TEST", "HEMORRHAGE")
        elif mode == "test-synthetic":
            folder = os.path.join("TEST", "CUBE")
        else:
            raise NameError("The specified dataset mode is not expected. Specify either train, val or test")

        images_root = os.path.join(root, folder)

        # save specifics
        self.spatial_size = spatial_size
        
        # save the augmentation functions, with identity in position 0
        self.augmentations = [IdentityD(keys=["image"])]
        if augmentations is not None:
            self.augmentations.extend(augmentations)

        # data multiplies with the number of augmentation functions
        self.data_multiplier = len(self.augmentations)  

        # get the names of all images
        image_names = sorted(os.listdir(images_root))

        self.resizing = Compose([
            ResizeD(keys=["image"], spatial_size=(self.spatial_size,) * 3),
        ])

        self.image_paths = []
        self.images = []

        for name in image_names:
            for aug in self.augmentations:
                path = os.path.join(images_root, name)
                self.image_paths.append(path)
                img = nib.load(path).get_fdata().astype(np.float32)
                data = {"image": img[np.newaxis, ...]}
                resized = self.resizing(data)
                augmented = aug(resized)
                self.images.append(augmented)

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        # load the image and label
        output = self.images[index]
        
        if "NORMAL" in self.image_paths[index]:
            output["label"] = 0
        else:
            output["label"] = 1
        
        # add scan number for debugging
        if "_" in Path(self.image_paths[index]).name:
            output["number"] = int(Path(self.image_paths[index]).name.split("-")[1].split('_')[0])
        else:
            output["number"] = int(Path(self.image_paths[index]).name.split("-")[1].split('.')[0])

        return output

# --------- THIS CODE CAN BE USED WHEN RAM IS TOO SMALL TO PRELOAD DATA --------------
# class Larynx_Data(Dataset):

#     def __init__(self, root, mode="train", augmentations=None, spatial_size=128):
#         # get the correct root paths
#         self.mode = mode
#         if mode == "train":
#             folder = "TRAIN"
#         elif mode == "val":
#             folder = "VAL"
#         elif mode == "test-normal":
#             folder = os.path.join("TEST", "NORMAL")
#         elif mode == "test-fracture":
#             folder = os.path.join("TEST", "FRACTURE")
#         elif mode == "test-hemorrhage":
#             folder = os.path.join("TEST", "HEMORRHAGE")
#         elif mode == "test-synthetic":
#             folder = os.path.join("TEST", "CUBE")
#         else:
#             raise NameError("The specified dataset mode is not expected. Specify either train, val or test")

#         images_root = os.path.join(root, folder)

#         # save specifics
#         self.spatial_size = spatial_size
        
#         # save the augmentation functions, with identity in position 0
#         self.augmentations = [IdentityD(keys=["image"])]
#         if augmentations is not None:
#             self.augmentations.extend(augmentations)

#         # data multiplies with the number of augmentation functions
#         self.data_multiplier = len(self.augmentations)  

#         # get the names of all images
#         image_names = sorted(os.listdir(images_root))

#         # save the complete paths to the individual images and labels
#         self.image_paths = [os.path.join(images_root, x) for x in image_names]

#     def __len__(self):
#         return len(self.image_paths) * self.data_multiplier

#     def __getitem__(self, index):
#         # find out if augmentation needed or not
#         augmentation_type = index // len(self.image_paths)
#         path_index = index % len(self.image_paths)

#         # load the image and label
#         path = self.image_paths[path_index]
#         data = {"image": path}
#         loading = Compose(
#             [
#                 LoadImageD(keys=["image"], reader="NibabelReader"),
#                 EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
#             ]
#         )
#         data = loading(data)

#         # resizing because 3D
#         resizing = Compose(
#             [
#                 ResizeD(keys=["image"],
#                         spatial_size=(self.spatial_size, self.spatial_size, self.spatial_size)),
#             ]
#         )
#         output = resizing(data)

#         # apply augmentation if needed
#         if augmentation_type > 0 and self.mode=='train':
#             augmentation = self.augmentations[augmentation_type]
#             output = augmentation(output)
        
#         if "NORMAL" in self.image_paths[path_index]:
#             output["label"] = 0
#         else:
#             output["label"] = 1
        
#         # add scan number for debugging
#         output["number"] = int(Path(self.image_paths[path_index]).name.split("-")[1].split('_')[0])
#         return output
