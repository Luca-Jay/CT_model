import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import LoadImage
from utils.visualization import viz_residual_heatmap
from lightning_modules.ae import AE

def load_ct_images(ct_path):
    loader = LoadImage(image_only=True)
    images = []
    for file in os.listdir(ct_path):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            image = loader(os.path.join(ct_path, file))
            images.append(image)
    return images

def compute_residual_maps(model, images):
    residual_maps = []
    for image in images:
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        reconstructions, _ = model(image_tensor)
        residual = torch.abs(reconstructions - image_tensor)
        residual_maps.append(residual.squeeze().detach().numpy())  # Remove batch and channel dimensions
    return residual_maps

def save_residual_maps(residual_maps, output_path):
    os.makedirs(output_path, exist_ok=True)
    for i, residual in enumerate(residual_maps):
        nifti_img = nib.Nifti1Image(residual, np.eye(4))
        nib.save(nifti_img, os.path.join(output_path, f'residual_map_{i}.nii.gz'))

def main(ct_path, output_path, model_path):
    model = AE.load_from_checkpoint(model_path, latent_size=512)  # Load your trained model
    model.eval()
    
    images = load_ct_images(ct_path)
    residual_maps = compute_residual_maps(model, images)
    save_residual_maps(residual_maps, output_path)

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Generate residual maps for CT images.")
    # parser.add_argument("ct_path", default="DATA\TEST\CUBE", type=str, help="Path to the directory containing CT images.")
    # parser.add_argument("output_path", default="DATA\TEST\CUBE", type=str, help="Path to the directory to save residual maps.")
    # parser.add_argument("model_path", default="OUTPUT\AE\checkpoints\epoch=69.ckpt", type=str, help="Path to the trained model checkpoint.")
    # args = parser.parse_args()

    # Hardcoded arguments
    ct_path = "DATA/TEST/CUBE"
    output_path = "DATA/TEST/CUBE"
    model_path = "OUTPUT/AE/checkpoints/epoch=69.ckpt"
    
    main(ct_path, output_path, model_path)
