import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import LoadImage
from utils.visualization import viz_residual_heatmap
from lightning_modules.ae import AE
from lightning_modules.vae import VAE
from lightning_modules.ae_msssim import AE_MSSSIM
from lightning_modules.vae_msssim import VAE_MSSSIM
from lightning_modules.ae_msssim_acai import AE_MSSSIM_ACAI
from lightning_modules.vae_msssim_acai import VAE_MSSSIM_ACAI
from lightning_modules.igd import IGD

from pytorch_msssim import ms_ssim

from torch.nn import functional as F



def load_ct_images(ct_path):
    loader = LoadImage(image_only=True)
    images = []
    for file in os.listdir(ct_path):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            image = loader(os.path.join(ct_path, file))
            images.append((image, file))
    return images

def compute_residual_maps(model, images, architecture):
    residuals = []
    for (image, path) in images:
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        
        reconstructions = model(image_tensor)[0]
        # loss = 1 - ms_ssim(image_tensor, reconstructions, data_range=1, size_average=True, win_size=7)
        loss = F.l1_loss(reconstructions, image_tensor, reduction='none')
        residuals.append((torch.mean(loss), path))  # Store loss and file path
    return residuals

def save_residual_maps(residual_maps, output_path):
    os.makedirs(output_path, exist_ok=True)
    for i, (residual, path) in enumerate(residual_maps):
        nifti_img = nib.Nifti1Image(residual, np.eye(4))
        nib.save(nifti_img, os.path.join(output_path, f'residual_map_{path}.nii.gz'))

def generate_residual_maps(ct_path, output_path, checkpoint, architecture, latent_size):
    # model
    rho = 0.15
    lambda_fool = 1.
    gamma = 0.2
    if architecture == 'AE':
        model = AE.load_from_checkpoint(checkpoint, latent_size=latent_size)

    elif architecture == 'AE_MSSSIM':
        model = AE_MSSSIM.load_from_checkpoint(checkpoint, latent_size=latent_size, rho=rho)

    elif architecture == 'VAE':
        model = VAE.load_from_checkpoint(checkpoint, latent_size=latent_size)

    elif architecture == 'VAE_MSSSIM':
        model = VAE_MSSSIM.load_from_checkpoint(checkpoint, latent_size=latent_size, rho=rho)

    elif architecture == 'AE_MSSSIM_ACAI':
        model = AE_MSSSIM_ACAI.load_from_checkpoint(checkpoint, latent_size=latent_size, rho=rho, lambda_fool=lambda_fool, gamma=gamma)

    elif architecture == 'VAE_MSSSIM_ACAI':
        model = VAE_MSSSIM_ACAI.load_from_checkpoint(checkpoint, latent_size=latent_size, rho=rho, lambda_fool=lambda_fool, gamma=gamma)

    elif architecture == 'IGD':
        model = IGD.load_from_checkpoint(checkpoint, latent_size=latent_size, rho=rho, lambda_fool=lambda_fool, gamma=gamma, map_location=torch.device('cuda:' + str(devices)))

    model.eval()
    
    images = load_ct_images(ct_path)
    losses = compute_residual_maps(model, images, architecture)
    for loss, path in losses:
        print(f"File: {path}, MS-SSIM Error: {loss}")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Generate residual maps for CT images.")
    # parser.add_argument("ct_path", default="DATA\TEST\CUBE", type=str, help="Path to the directory containing CT images.")
    # parser.add_argument("output_path", default="DATA\TEST\CUBE", type=str, help="Path to the directory to save residual maps.")
    # parser.add_argument("model_path", default="OUTPUT\AE\checkpoints\epoch=69.ckpt", type=str, help="Path to the trained model checkpoint.")
    # args = parser.parse_args()

    # Hardcoded arguments
    ct_path = "/workspace/project-data/CT_model/DATA/TIGHT/TEST/NORMAL"
    output_path = "/workspace/project-data/CT_model/OUTPUT/RESIDUALS/"
    checkpoint = "/workspace/project-data/CT_model/OUTPUT/TIGHT/AE_MSSSIM_ACAI/checkpoints/04-03 20:38 - BS:16, EP: 200, LS:1024, AUG: 8, CV: [(300, 1500)]/epoch=159.ckpt"
    architecture = "AE_MSSSIM_ACAI"
    latent_size=  1024
    
    generate_residual_maps(ct_path, output_path, checkpoint, architecture, latent_size)
