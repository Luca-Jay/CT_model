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
    architectures = ['VAE_MSSSIM_ACAI']
    batch_size = 16
    epochs = 100
    latent_size = 1024
    spatial_size = 128
    accelerator = 'gpu' if num_gpus() > 0 else 'cpu'
    devices = num_gpus() if num_gpus() > 0 else 1
    dataset_dir = '/workspace/project-data/CT_model/DATA/TIGHT'
    output_dir = '/workspace/project-data/CT_model/OUTPUT/TIGHT'
    mean_map = False

    # Number of randomized augmentations per image
    number_of_augmentations = [2]

    # Define spatial & intensity options
    augmentations = [
        RandFlipD(keys=["image"], spatial_axis=0, prob=1.0),
        RandRotateD(keys=["image"], range_x=0.1, prob=1.0),
        RandZoomD(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=1.0),
        RandAffineD(keys=["image"], rotate_range=(0.1, 0, 0), translate_range=(5, 5, 5), prob=1.0),
    ]

    intensity_aug = [
        RandShiftIntensityD(keys=["image"], offsets=0.1, prob=1.0),
        RandScaleIntensityD(keys=["image"], factors=0.1, prob=1.0),
        RandBiasFieldD(keys=["image"], prob=1.0),
        RandGaussianNoiseD(keys=["image"], std=0.01, prob=1.0),
        RandAdjustContrastD(keys=["image"], gamma=(0.9, 1.1), prob=1.0),
        RandHistogramShiftD(keys=["image"], prob=1.0),
    ]

    

    clipping_values = [(300 , 1500)]  # Default HU ranges for channels , (500 , 2000), (50, 400)

    for architecture in architectures:
        # Compose N randomized augmentation pipelines
        for N in number_of_augmentations:
            for n in range(N):
                intensity = intensity_aug[n]
                augmentations.append(intensity)
            for clipping_value in clipping_values:
                print(f"Training {architecture} model...")
                train_model(batch_size, epochs, architecture, latent_size, spatial_size, accelerator, devices, dataset_dir, output_dir, augmentations, [clipping_value])
                
                # checkpoint_dir = os.path.join(output_dir, architecture, 'checkpoints')
                # checkpoint = get_latest_checkpoint(checkpoint_dir)
                
                # print(f"Testing {architecture} model with checkpoint {checkpoint}...")
                # test_model(batch_size, checkpoint, architecture, mean_map, dataset_dir, accelerator, devices, latent_size, clipping_values)

if __name__ == '__main__':
    main()
