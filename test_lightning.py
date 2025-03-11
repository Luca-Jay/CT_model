import pytorch_lightning as pl
from argparse import ArgumentParser
import pandas as pd
import torch
from pathlib import Path
import seaborn as sns

from lightning_modules.ae import AE
from lightning_modules.vae import VAE
from lightning_modules.ae_msssim import AE_MSSSIM
from lightning_modules.vae_msssim import VAE_MSSSIM
from lightning_modules.ae_msssim_acai import AE_MSSSIM_ACAI
from lightning_modules.vae_msssim_acai import VAE_MSSSIM_ACAI
from lightning_modules.igd import IGD

import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.larynx_data import Larynx_Data
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import get_best_threshold, Collector
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from utils.visualization import viz_pr_curve, viz_confusion_matrix

def test_model(batch_size, checkpoint, architecture, mean_map, dataset_dir, accelerator, devices, latent_size):
    pl.seed_everything(42, workers=True)

    # initialize output directory
    output_dir = Path(os.path.join(os.path.dirname(os.path.dirname(checkpoint)), "testing" if not mean_map else 'testing-mask'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # data
    data_dir = dataset_dir
    dataset_normal = Larynx_Data(root=data_dir, mode="test-normal")
    dataset_HEM = Larynx_Data(root=data_dir, mode="test-hemorrhage")
    dataset_FRAC = Larynx_Data(root=data_dir, mode="test-fracture")
    dataset_SYN = Larynx_Data(root=data_dir, mode="test-synthetic")
    abnormal_datasets = {"SYN": dataset_SYN}

    train_dataset = Larynx_Data(root=data_dir, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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

    # metrics containers for DataFrame
    metrics = []
    average_scores = []
    pr_curves_rec = []
    pr_curves_feat = []
    pr_curves_combined = []

    # calculate metrics for all diseases
    for disease, dataset in abnormal_datasets.items():

        # initialize the containers for predictions and targets
        if accelerator is 'gpu':
            model = model.cuda(device=torch.device('cuda:' + str(devices)))
        model.codes = Collector(500)
        model.rec_preds = Collector(500)
        model.feat_preds = Collector(500)
        model.combined_preds = Collector(500)
        model.targets = Collector(500)

        # initialize the mean_map if necessary
        if mean_map:
            model.mean_map = torch.load("./", map_location=model.device).unsqueeze(0)

        # logger
        exp_name = "abnormality_" + disease
        print("Started: " + exp_name)
        logger = TensorBoardLogger(save_dir=output_dir, name=exp_name)
        test_loader_normal = DataLoader(dataset_normal, shuffle=False, batch_size=batch_size, num_workers=4)
        test_loader_abnormal = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)

        # test, test, test
        trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=logger)
        
        model.visualize_testing = False
        trainer.test(model, dataloaders=[test_loader_normal, test_loader_abnormal])

        # collect predictions and targets
        codes = model.codes.get_all()
        rec_preds = model.rec_preds.get_all()
        feat_preds = model.feat_preds.get_all()
        combined_preds = model.combined_preds.get_all()
        targets = model.targets.get_all()
        
        # visualize the embedding in projector to see if we get two clusters
        logger.experiment.add_embedding(codes, targets, tag="Normal vs Abnormal 3D")
        
        # visualize the embeddings as image
        logger.experiment.add_image("Embeddings visualization", codes.unsqueeze(0))


        #####################
        # ON FEATURE SCORES #
        #####################
        
        # create PR curve on feature scores
        feat_curve = precision_recall_curve(targets.cpu().numpy(), feat_preds.cpu().numpy())
        pr_curves_feat.append(feat_curve)

        # pick the best threshold (f1 score-wise)
        best_f_feat, best_thr_feat = get_best_threshold(feat_curve)
        
        # compute and plot confusion matrix for best threshold
        cf_mat = confusion_matrix(targets.cpu().numpy(), (feat_preds > best_thr_feat).cpu().numpy())
        viz_confusion_matrix(cf_mat, "CF Matrix, Features/f-score: " + str(round(best_f_feat, 4)) 
                                                    + " | thr: " + str(round(best_thr_feat, 4)) 
                                                    + " | disease: " + disease, logger.experiment)

        ###################
        # ON IMAGE SCORES #
        ###################
        
        # create PR curve on image scores
        rec_curve = precision_recall_curve(targets.cpu().numpy(), rec_preds.cpu().numpy())
        pr_curves_rec.append(rec_curve)

        # pick the best threshold (f1 score-wise)
        best_f_rec, best_thr_rec = get_best_threshold(rec_curve)
        
        # compute and plot confusion matrix for best threshold
        cf_mat = confusion_matrix(targets.cpu().numpy(), (rec_preds > best_thr_rec).cpu().numpy())
        viz_confusion_matrix(cf_mat, "CF Matrix, Reconstr./f-score: " + str(round(best_f_rec, 4)) 
                                                    + " | thr: " + str(round(best_thr_rec, 4)) 
                                                    + " | disease: " + disease, logger.experiment)

        # append metrics for both scores
        metrics.append([disease, architecture, roc_auc_score(targets.cpu().numpy(), rec_preds.cpu().numpy()),
                                                    roc_auc_score(targets.cpu().numpy(), feat_preds.cpu().numpy()),
                                                    roc_auc_score(targets.cpu().numpy(), combined_preds.cpu().numpy()),
                                                    average_precision_score(targets.cpu().numpy(), rec_preds.cpu().numpy()),
                                                    average_precision_score(targets.cpu().numpy(), feat_preds.cpu().numpy()),
                                                    average_precision_score(targets.cpu().numpy(), combined_preds.cpu().numpy())])  
        
        # append first the average scores for healthy samples
        if disease == 'SYN':
            for i in range(rec_preds[targets==0].shape[0]):
                average_scores.append(["NORMAL", architecture, rec_preds[targets==0][i].item(),
                                                                    feat_preds[targets==0][i].item(),
                                                                    combined_preds[targets==0][i].item()])  
        
        # append average scores for unhealthy
        for i in range(rec_preds[targets==1].shape[0]):
            average_scores.append([disease, architecture, rec_preds[targets==1][i].item(),
                                                            feat_preds[targets==1][i].item(),
                                                            combined_preds[targets==1][i].item()])
        

        ##########################
        # TEST FOR VISUALIZATION #
        ##########################
        model.visualize_testing = True
        model.thr_feat = best_thr_feat
        model.thr_rec = best_thr_rec

        trainer.test(model, dataloaders=[test_loader_normal, test_loader_abnormal])


    # visualize the PR curves
    viz_pr_curve(pr_curves_feat, "PR Curve, Feature | " + architecture, logger.experiment)    
    viz_pr_curve(pr_curves_rec, "PR Curve, Reconstr. | " + architecture, logger.experiment)                                         
    viz_pr_curve(pr_curves_combined, "PR Curve, Combined | " + architecture, logger.experiment)

    # get output filename for metrics xlsx
    output_filename = architecture + "-metrics.xlsx"

    if mean_map:
        output_filename = "masked-" + output_filename

    # accumulate metrics in DataFrame and save it
    metrics_df = pd.DataFrame(metrics, columns=["Abnormality", "Method", "AUROC(I)", "AUROC(F)", "AUROC(C)", "AvgPrec(I)", "AvgPrec(F)", "AvgPrec(C)"])
    metrics_df.to_excel(output_dir / output_filename, float_format="%.4f")  

    # accumulate average scores in DataFrame and save it
    avg_score_df = pd.DataFrame(average_scores, columns=['Abnormality', 'Method', 'I score', 'F score', 'C score'])
    ax = sns.violinplot(x='I score', y='Abnormality', data=avg_score_df, density_norm='width', palette='Set3')
    ax.set(xlim=(0.01, 0.12))
    logger.experiment.add_figure(tag='Violin for I score NEW, width', figure=ax.get_figure())
    ax = sns.violinplot(x='F score', y='Abnormality', data=avg_score_df, density_norm='count')
    logger.experiment.add_figure(tag='Violin for F score, count', figure=ax.get_figure())
    ax = sns.violinplot(x='C score', y='Abnormality', data=avg_score_df, density_norm='count')
    logger.experiment.add_figure(tag='Violin for C score, count', figure=ax.get_figure())

    # get output filename for metrics xlsx
    output_filename = architecture + "-average_scores.xlsx"
    if mean_map:
        output_filename = "masked-" + output_filename
    avg_score_df.to_excel(output_dir / output_filename, float_format="%.4f")

if __name__ == '__main__':
    # Hardcoded arguments for testing
    batch_size = 2
    checkpoint = 'OUTPUT\AE\checkpoints\epoch=199.ckpt'
    architecture = 'AE'
    mean_map = False
    accelerator = 'cpu'
    devices = None
    dataset_dir = 'DATA'

    test_model(batch_size, checkpoint, architecture, mean_map, accelerator, devices, dataset_dir)
