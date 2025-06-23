# Trauma Tracer CT Model
This is the model for the TraumaTracer project. The code belongs to the thesis "AI for Forensic Radiologists: An AI Pipeline for Laryngeal Abnormality Detection" by Luca Pamboer. 

This project provides code for training and evaluating deep learning models to detect trauma (e.g., strangulation, asphyxia) in CT scans. It includes utilities for data preparation, model training, testing, and visualization. The code is based on "CTFSH: Full Head CT Anomaly Detection with  Unsupervised Learning" by Guerra et al.

## Project Purpose

The goal is to automate trauma detection in CT scans using neural networks. The code supports data splitting, model training (with various architectures), and evaluation on different trauma categories.

## Folder Structure

Your data and project folders should be organized as follows:

```
CT_model/
│
├── DATA/
│    ├── TRAIN/
│    ├── VAL/
│    └── TEST/
│        ├── NORMAL/
│        ├── STRANGULATION/
│        ├── ASPHYXIA/
│        └── ... (other categories as needed)

```

- **TRAIN/**: Training data ("natural" cases).
- **VAL/**: Validation data.
- **TEST/**: Test data, split into subfolders by trauma type (e.g., NORMAL, STRANGULATION, ASPHYXIA).

## Data Preparation

### 1. Splitting Data

Use `split_data.py` to organize your raw CT scan files and metadata into the required folder structure.

**Usage:**

- Set the paths at the top of `split_data.py`:
  - `data_dir`: Directory with your raw `.nii.gz` CT scans.
  - `decedentsfile`: Path to your Excel metadata file.
  - `output_dir`: Where to create the TRAIN/VAL/TEST folders.

- Run the script:
  ```bash
  python split_data.py
  ```

This will:
- Read metadata and CT files.
- Split "natural" cases into TRAIN/VAL/TEST.
- Place trauma cases (e.g., strangulation, asphyxia) into their respective TEST subfolders.
- Rename files for trauma cases to indicate their type.

### 2. Model Training

Train a model using `train_lightning.py`.

**Example usage:**
```bash
python train_lightning.py --batch_size 8 --epochs 100 --architecture AE_MSSSIM_ACAI --latent_size 1024 --spatial_size 128 --accelerator gpu --device 0 --dataset_dir DATA/TIGHT_ALL_HANGING --output_dir OUTPUT/HANGING
```

**Key arguments:**
- `--architecture`: Model type (`AE`, `AE_MSSSIM`, `AE_MSSSIM_ACAI`, `VAE`, etc.).
- `--latent_size`: Latent vector size (e.g., 1024).
- `--spatial_size`: Input volume size (e.g., 128).
- `--dataset_dir`: Path to your prepared data directory.
- `--output_dir`: Where to save logs and checkpoints.

Augmentations and clipping values can be customized in the script.

### 3. Model Testing

Evaluate a trained model using `test_lightning.py`.

**Example usage:**
```bash
python test_lightning.py --batch_size 8 --checkpoint OUTPUT/AE_MSSSIM_ACAI/checkpoints/epoch=99.ckpt --architecture AE_MSSSIM_ACAI --dataset_dir DATA/ --latent_size 1024
```

- Outputs metrics (AUROC, average precision, etc.) and visualizations to the output directory.
- Evaluates on both normal and trauma test sets.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- MONAI
- NumPy, pandas, scikit-learn, seaborn, nibabel, tqdm

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Additional Utilities

- `utils/visualization.py`: Visualization functions for training/testing.
- `utils/generate_residual_maps.py`: Generate and save residual maps from trained models.

## Notes

- Ensure your CT data and metadata are correctly formatted and matched.
- Adjust hyperparameters and augmentations as needed for your experiments.
- For custom data splits or trauma categories, modify `split_data.py` accordingly.

## License

This project is for research purposes. Contact the author for licensing information.

