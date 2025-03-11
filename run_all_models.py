import os
from train_lightning import train_model
from test_lightning import test_model
import torch
from monai.transforms import RandRotateD, ScaleIntensityD, RandZoomD, RandFlipD
import numpy as np

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
    latent_size = 512
    spatial_size = 128
    accelerator = 'gpu' if num_gpus() > 1 else 'cpu'
    devices = num_gpus() if num_gpus() > 1 else 1
    dataset_dir = 'DATA'
    output_dir = 'OUTPUT'
    mean_map = False

    augmentations = [
        ScaleIntensityD(keys=["image"]),
        RandFlipD(keys=["image"], spatial_axis=0),
        RandRotateD(keys=["image"], range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlipD(keys=["image"], spatial_axis=0, prob=0.5),
        RandZoomD(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5)
    ]

    for architecture in architectures:
        print(f"Training {architecture} model...")
        train_model(batch_size, epochs, architecture, latent_size, spatial_size, accelerator, devices, dataset_dir, output_dir, augmentations)
        
        checkpoint_dir = os.path.join(output_dir, architecture, 'checkpoints')
        checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        print(f"Testing {architecture} model with checkpoint {checkpoint}...")
        test_model(batch_size, checkpoint, architecture, mean_map, dataset_dir, accelerator, devices, latent_size)

if __name__ == '__main__':
    main()
