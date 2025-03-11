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

def train_model(batch_size, epochs, architecture, latent_size, spatial_size, accelerator, devices, dataset_dir, output_dir, augmentations=None):
    pl.seed_everything(42, workers=True)

    # data
    dataset_root = dataset_dir
    datamodule = Larynx_DataModule(data_dir=dataset_root, batch_size=batch_size, spatial_size=spatial_size, augmentations=augmentations)

    rho = 0.15
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
    train_logger = TensorBoardLogger(save_dir=root_log_dir, name="pretraining")
    
    # create checkpoint callback
    checkpoint_dir = os.path.join(root_log_dir, "checkpoints") 
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        save_last=True,
        every_n_epochs=10,
    )
    
    # create trainer object
    trainer = pl.Trainer(accelerator=accelerator,
                            devices=devices, 
                            logger=train_logger, 
                            fast_dev_run=False,
                            num_sanity_val_steps=0,
                            log_every_n_steps=20,
                            callbacks=[checkpoint_callback],
                            max_epochs=epochs,
                            overfit_batches=1)
    trainer.fit(model, datamodule)

# entry point
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--architecture', default='IGD', choices=['AE', 'AE_MSSSIM', 'AE_MSSSIM_ACAI', 'VAE', 'VAE_MSSSIM', 'VAE_MSSSIM_ACAI', 'IGD'], type=str)
    parser.add_argument('--latent_size', default=512, choices=[256, 512, 1024], type=int)
    parser.add_argument('--spatial_size', default=128, choices=[64, 128], type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--dataset_dir', default='.', type=str)
    parser.add_argument('--output_dir', default='.', type=str)
    args = parser.parse_args()

    train_model(args.batch_size, args.epochs, args.architecture, args.latent_size, args.spatial_size, args.gpu, args.dataset_dir, args.output_dir)