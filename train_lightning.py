import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.larynx_data_module import Larynx_DataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning_modules.ae_msssim_acai import AE_MSSSIM_ACAI
from lightning_modules.igd import IGD
from lightning_modules.vae_msssim_acai import VAE_MSSSIM_ACAI
from lightning_modules.ae import AE
from lightning_modules.vae import VAE
from lightning_modules.ae_msssim import AE_MSSSIM
from lightning_modules.vae_msssim import VAE_MSSSIM
from datetime import datetime

def train_model(batch_size, epochs, architecture, latent_size, spatial_size, accelerator, devices, dataset_dir, output_dir, augmentations=None, clipping_values=None, rho=0.15):
    pl.seed_everything(42, workers=True)

    # data
    dataset_root = dataset_dir
    datamodule = Larynx_DataModule(data_dir=dataset_root, batch_size=batch_size, spatial_size=spatial_size, augmentations=augmentations, clipping_values=clipping_values)

    # rho = 0.15
    lambda_fool = 0.1
    gamma = 0.2
    # build the model
    if architecture == "AE":
        model = AE(latent_size)

    elif architecture == "VAE":
        model = VAE(latent_size)
    
    elif architecture == 'AE_MSSSIM':
        model = AE_MSSSIM(latent_size, rho)
    
    elif architecture == 'VAE_MSSSIM':
        model = VAE_MSSSIM(latent_size, rho)
    
    elif architecture == 'AE_MSSSIM_ACAI':
        model = AE_MSSSIM_ACAI(latent_size, rho, lambda_fool, gamma)
    
    elif architecture == 'VAE_MSSSIM_ACAI':
        model = VAE_MSSSIM_ACAI(latent_size, rho, lambda_fool, gamma)
    
    elif architecture == 'IGD':
        model = IGD(latent_size, rho, lambda_fool, gamma)
        

    # choose gpu and logger
    experiment_name = architecture
    root_log_dir = os.path.join(output_dir, experiment_name)
    now = datetime.now().strftime("%m-%d %H:%M")
    version_name = f"{now} - BS:{batch_size}, EP: {epochs}, LS:{latent_size}, AUG: {len(augmentations) if augmentations else 0}, CV: {clipping_values}, rho: {rho}"
    train_logger = TensorBoardLogger(save_dir=root_log_dir, name="pretraining", version=version_name)

    # create checkpoint callback
    checkpoint_dir = os.path.join(root_log_dir, "checkpoints", version_name) 
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        save_last=False,
        every_n_epochs=10,
    )
    
    # create trainer object
    trainer = pl.Trainer(accelerator=accelerator,
                            devices=devices, 
                            logger=train_logger, 
                            fast_dev_run=False,
                            num_sanity_val_steps=1,
                            log_every_n_steps=20,
                            callbacks=[checkpoint_callback],
                            max_epochs=epochs,
                            #enable_checkpointing=False
                        )


    trainer.fit(model, datamodule)

# entry point
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--architecture', default='AE_MSSSIM_ACAI', choices=['AE', 'AE_MSSSIM', 'AE_MSSSIM_ACAI', 'VAE', 'VAE_MSSSIM', 'VAE_MSSSIM_ACAI', 'IGD'], type=str)
    parser.add_argument('--latent_size', default=1024, choices=[256, 512, 1024, 2048], type=int)
    parser.add_argument('--spatial_size', default=128, choices=[64, 128], type=int)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu', 'cpu'], type=str)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--dataset_dir', default='/workspace/project-data/CT_model/DATA/TIGHT_ALL_HANGING', type=str)
    parser.add_argument('--output_dir', default='/workspace/project-data/CT_model/OUTPUT/HANGING', type=str)
    parser.add_argument('--rho', default=0.15, type=int)
    args = parser.parse_args()

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

    train_model(args.batch_size, args.epochs, args.architecture, args.latent_size, args.spatial_size, args.accelerator, args.device, args.dataset_dir, args.output_dir, spatial+intensity_aug, [(3000, 50)], args.rho)
