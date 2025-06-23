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

### Code to generate residual maps from CT scans using a trained model ###

def load_ct_images(ct_path):
    loader = LoadImage(image_only=True)
    images = []
    for file in os.listdir(ct_path):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            image = loader(os.path.join(ct_path, file))
            images.append((image, file))
    return images

def compute_residual_maps(model, images):
    residual_maps = []
    for (image, path) in images:
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        reconstructions = model(image_tensor.to(device='cuda'))[0].cpu()
        
        residual = torch.abs(reconstructions - image_tensor)
        residual = residual * 1000  # Scale residual values to range 0 to 1000
        residual_maps.append((residual.cpu().squeeze().detach().numpy(), path))  # Remove batch and channel dimensions
    return residual_maps

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
    residual_maps = compute_residual_maps(model, images)
    save_residual_maps(residual_maps, output_path)

if __name__ == "__main__":
    ct_path = "Data/Case--001/CT-SCAN-ST"
    output_path = "Interview data/"
    checkpoint = "AE_MSSSIM_ACAI/checkpoints/epoch=199.ckpt"
    architecture = "AE_MSSSIM_ACAI"
    latent_size=  1024
    
    generate_residual_maps(ct_path, output_path, checkpoint, architecture, latent_size)
