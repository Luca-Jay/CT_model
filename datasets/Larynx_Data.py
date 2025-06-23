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


# class Larynx_Data(Dataset):

#     def __init__(self, root, mode="train", augmentations=None, spatial_size=128, clipping_values=None):
#         # get the correct root paths
#         self.mode = mode
#         if mode == "train":
#             folder = "TRAIN"
#         elif mode == "val":
#             folder = "VAL"
#         elif mode == "test-normal":
#             folder = os.path.join("TEST", "NORMAL")
#         elif mode == "test-strangulation":
#             folder = os.path.join("TEST", "STRANGULATION")
#         elif mode == "test-hemorrhage":
#             folder = os.path.join("TEST", "HEMORRHAGE")
#         elif mode == "test-synthetic10":
#             folder = os.path.join("TEST", "CUBE10")
#         elif mode == "test-synthetic15":
#             folder = os.path.join("TEST", "CUBE15")
#         elif mode == "test-synthetic30":
#             folder = os.path.join("TEST", "CUBE30")
#         elif mode == "test-asphyxia":
#             folder = os.path.join("TEST", "ASPHYXIA")
#         else:
#             raise NameError("The specified dataset mode is not expected. Specify either train, val or test")

#         images_root = os.path.join(root, folder)

#         # save specifics
#         self.spatial_size = spatial_size
#         self.clipping_values = clipping_values if clipping_values else [(None, None)]
        
#         # save the augmentation functions, with identity in position 0
#         self.augmentations = []
#         if augmentations is not None:
#             self.augmentations.extend(augmentations)

#         # data multiplies with the number of augmentation functions
#         self.data_multiplier = len(self.augmentations)  

#         # get the names of all images
#         image_names = sorted(os.listdir(images_root))

#         self.resizing = Compose([
#             ResizeD(keys=["image"], spatial_size=(self.spatial_size,) * 3),
#         ])

#         self.image_paths = []
#         self.images = []

#         for name in image_names:
#             path = os.path.join(images_root, name)
#             img = nib.load(path).get_fdata().astype(np.float32)
#             channels = []
#             for center, width in self.clipping_values:
#                 clamped_img = window(img, center, width) if center is not None and width is not None else img
#                 normalized_img = (clamped_img - clamped_img.min()) / (clamped_img.max() - clamped_img.min())
#                 channels.append(normalized_img)
#             stacked_img = np.stack(channels, axis=0)  # Stack channels
#             data = {"image": stacked_img}
#             data = self.resizing(data)

#             if "NORMAL" in path or "TRAIN" in path:
#                 data["label"] = 0
#             else:
#                 data["label"] = 1
            
#             # add scan number for debugging
#             if "_" in Path(path).name:
#                 data["number"] = int(Path(path).name.split("-")[1].split('_')[0])
#             else:
#                 data["number"] = int(Path(path).name.split("-")[1].split('.')[0])

#             self.images.append(data)
#             if mode == 'train':
#                 for aug in self.augmentations:
#                     self.image_paths.append(path)
#                     augmented = aug(data)
#                     self.images.append(augmented)

#     def __len__(self):
#         return len(self.images)

#     def get_case_numbers(self):
#         return [image['number'] for image in self.images]


#     def __getitem__(self, index):
#         # load the image and label
#         image = self.images[index]
        
#         # if "NORMAL" in self.image_paths[index] or "TRAIN" in self.image_paths[index]:
#         #     output["label"] = 0
#         # else:
#         #     output["label"] = 1
        
#         # # add scan number for debugging
#         # if "_" in Path(self.image_paths[index]).name:
#         #     output["number"] = int(Path(self.image_paths[index]).name.split("-")[1].split('_')[0])
#         # else:
#         #     output["number"] = int(Path(self.image_paths[index]).name.split("-")[1].split('.')[0])

#         return image

# # # --------- THIS CODE CAN BE USED WHEN RAM IS TOO SMALL TO PRELOAD DATA --------------
class Larynx_Data(Dataset):

    def __init__(self, root, mode="train", augmentations=None, spatial_size=128, clipping_values=[(None, None)]):
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
        elif mode == "test-hanging":
            folder = os.path.join("TEST", "HANGING")
        elif mode == "test-synthetic10":
            folder = os.path.join("TEST", "CUBE10")
        elif mode == "test-synthetic15":
            folder = os.path.join("TEST", "CUBE15")
        elif mode == "test-synthetic30":
            folder = os.path.join("TEST", "CUBE30")
        elif mode == "test-asphyxia":
            folder = os.path.join("TEST", "ASPHYXIA")
        else:
            raise NameError("The specified dataset mode is not expected. Specify either train, val or test")

        images_root = os.path.join(root, folder)

        # save specifics
        self.spatial_size = spatial_size
        self.clipping_values = clipping_values if clipping_values else [(None, None)]
        
        # save the augmentation functions, with identity in position 0
        self.augmentations = augmentations
        # data multiplies with the number of augmentation functions
        self.data_multiplier = len(self.augmentations) if augmentations else 1

        # get the names of all images
        image_names = sorted(os.listdir(images_root))

        self.resizing = Compose([
            ResizeD(keys=["image"], spatial_size=(self.spatial_size,) * 3),
        ])

        self.image_paths = []
        self.images = []

        for name in image_names:
            path = os.path.join(images_root, name)
            img = nib.load(path).get_fdata().astype(np.float32)
            channels = []
            for center, width in self.clipping_values:
                clamped_img = window(img, center, width) if center is not None and width is not None else img
                normalized_img = (clamped_img - clamped_img.min()) / (clamped_img.max() - clamped_img.min())
                channels.append(normalized_img)
            stacked_img = np.stack(channels, axis=0)  # Stack channels
            data = {"image": stacked_img}
            data = self.resizing(data)


            if "NORMAL" in path or "TRAIN" in path:
                data["label"] = 0
            else:
                data["label"] = 1
            
            # add scan number for debugging
            if "_" in Path(path).name:
                data["number"] = int(Path(path).name.split("-")[1].split('_')[0])
            else:
                data["number"] = int(Path(path).name.split("-")[1].split('.')[0])

            self.images.append(data)
            self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths) * self.data_multiplier

    def get_case_numbers(self):
        return self.image_paths

    def __getitem__(self, index):
        # find out if augmentation needed or not
        augmentation_type = index // len(self.image_paths)
        path_index = index % len(self.image_paths)

        # load the image and label
        data = self.images[path_index]

        # apply augmentation if needed
        if augmentation_type > 0 and self.mode=='train':
            augmentation = self.augmentations[augmentation_type]
            data = augmentation(data)
        
        
        return data
