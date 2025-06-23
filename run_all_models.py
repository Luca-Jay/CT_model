import os
from train_lightning import train_model
from test_lightning import test_model
import torch
from monai.transforms import (
    Compose,
    RandFlipD,
    RandRotateD,
    RandZoomD,
    RandAffineD,
    RandBiasFieldD,
    RandShiftIntensityD,
    RandScaleIntensityD,
    RandGaussianNoiseD,
    RandAdjustContrastD,
    RandHistogramShiftD,
    IdentityD,
)
import numpy as np
import random

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('last-v') and f.endswith('.ckpt')]
    if not checkpoints:
        # Check for a checkpoint named 'last.ckpt'
        if 'last.ckpt' in os.listdir(checkpoint_dir):
            return os.path.join(checkpoint_dir, 'last.ckpt')
        raise FileNotFoundError("No checkpoint files found in the directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('v')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def main():
    # architectures = ['AE', 'VAE', 'AE_MSSSIM', 'VAE_MSSSIM', 'AE_MSSSIM_ACAI', 'VAE_MSSSIM_ACAI', 'IGD']
    architectures = ['AE_MSSSIM_ACAI']
    batch_size = 8
    epochs = 100
    latent_size = 1024
    spatial_size = 128
    accelerator = 'gpu' if num_gpus() > 0 else 'cpu'
    devices = num_gpus() if num_gpus() > 0 else 1
    dataset_dir = '/workspace/project-data/CT_model/DATA/TIGHT_ALL'
    output_dir = '/workspace/project-data/CT_model/OUTPUT/AUGMENTATION_newmethod'
    mean_map = False
    use_augmentations = True

    # Define spatial & intensity options
    spatial = [
        RandFlipD(keys=["image"], spatial_axis=[0], prob=1),
        #RandRotateD(keys=["image"], range_x=0.05, range_y=0.05, range_z=0.05, prob=1),
        RandZoomD(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=1),
        RandAffineD(keys=["image"], rotate_range=(0.5, 0.05, 0.05), translate_range=(5, 5, 5), scale_range=(0.05, 0.05, 0.05), prob=1),
    ]

    intensity_aug = [
        RandShiftIntensityD(keys=["image"], offsets=0.1, prob=1),               
        RandScaleIntensityD(keys=["image"], factors=0.3, prob=1),               
        # RandGaussianNoiseD(keys=["image"], std=0.03, prob=1),                   
        # RandAdjustContrastD(keys=["image"], gamma=(0.7, 1.4), prob=1),          
        # RandHistogramShiftD(keys=["image"], num_control_points=8, prob=1),      
    ]

    rhos = [0.05,0.10,0.25]

    clipping_values = [(300, 1500)]  # Default HU ranges for channels , (500 , 2000), (50, 400)

    for architecture in architectures:
        if use_augmentations:
            augmentations = spatial
        else:
            augmentations = None
        for clipping_value in clipping_values:
            print(f"Training {architecture} model...")
            train_model(batch_size, epochs, architecture, latent_size, spatial_size, accelerator, devices, dataset_dir, output_dir, augmentations, [clipping_value])

    # checkpoint_paths =[
    #     "CT_model/OUTPUT/TIGHT_ALL_AUGMENTATIONS_CLEAN/AE/checkpoints/05-01 11:43 - BS:16, EP: 200, LS:1024, AUG: 9, CV: [(300, 1500)]/epoch=199.ckpt",
    #     "CT_model/OUTPUT/TIGHT_ALL_AUGMENTATIONS_CLEAN/VAE/checkpoints/05-01 12:28 - BS:16, EP: 200, LS:1024, AUG: 9, CV: [(300, 1500)]/epoch=199.ckpt",
    #     "CT_model/OUTPUT/TIGHT_ALL_AUGMENTATIONS_CLEAN/AE_MSSSIM/checkpoints/05-01 07:54 - BS:16, EP: 200, LS:1024, AUG: 9, CV: [(300, 1500)]/epoch=89.ckpt",
    #     "CT_model/OUTPUT/TIGHT_ALL_AUGMENTATIONS_CLEAN/VAE_MSSSIM/checkpoints/05-01 13:12 - BS:16, EP: 200, LS:1024, AUG: 9, CV: [(300, 1500)]/epoch=199.ckpt",
    #     "CT_model/OUTPUT/TIGHT_ALL_AUGMENTATIONS_CLEAN/AE_MSSSIM_ACAI/checkpoints/05-01 08:32 - BS:16, EP: 200, LS:1024, AUG: 9, CV: [(300, 1500)]/epoch=199.ckpt",
    #     "CT_model/OUTPUT/TIGHT_ALL_AUGMENTATIONS_CLEAN/VAE_MSSSIM_ACAI/checkpoints/05-01 14:33 - BS:16, EP: 200, LS:1024, AUG: 9, CV: [(300, 1500)]/epoch=199.ckpt"
    # ]

    # for i in range(0,len(checkpoint_paths)):
    #     architecture = architectures[i]
    #     checkpoint_path = checkpoint_paths[i]
    #     test_model(batch_size, checkpoint_path, architecture, False, dataset_dir, "cpu", 1, latent_size, clipping_values)

if __name__ == '__main__':
    main()
