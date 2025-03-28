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
    architectures = ['AE']
    batch_size = 2
    epochs = 100
    latent_size = 1024
    spatial_size = 128
    accelerator = 'gpu' if num_gpus() > 0 else 'cpu'
    devices = num_gpus() if num_gpus() > 0 else 1
    dataset_dir = '/workspace/project-data/CT_model/DATA/TIGHT'
    output_dir = '/workspace/project-data/CT_model/OUTPUT/TIGHT'
    mean_map = False

    # Number of randomized augmentations per image
    N = 4

    # Identity for original image
    precomputed_augmentations = [IdentityD(keys=["image"])]

    # Define spatial & intensity options
    spatial_aug = [
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

    # Compose N randomized augmentation pipelines
    
    for _ in range(N):
        spatial = random.choice(spatial_aug)
        intensity = random.choice(intensity_aug)
        precomputed_augmentations.append(Compose([spatial, intensity]))

    hu_values = [(None, None), (0, 500), (200, 800)]  # Default HU ranges for channels

    for architecture in architectures:
        print(f"Training {architecture} model...")
        train_model(batch_size, epochs, architecture, latent_size, spatial_size, accelerator, devices, dataset_dir, output_dir, precomputed_augmentations, hu_values)
        
        checkpoint_dir = os.path.join(output_dir, architecture, 'checkpoints')
        checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        print(f"Testing {architecture} model with checkpoint {checkpoint}...")
        test_model(batch_size, checkpoint, architecture, mean_map, dataset_dir, accelerator, devices, latent_size, hu_values)

if __name__ == '__main__':
    main()
