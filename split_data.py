import os
import random
import shutil
from pathlib import Path

# Set the seed for reproducibility
random.seed(42)

# Define the paths
data_dir = '../CT_preprocessing/PREPROCESSED_CT_SCANS'
output_dir = 'DATA'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create output directories
train_dir = os.path.join(output_dir, 'TRAIN')
val_dir = os.path.join(output_dir, 'VAL')
test_dir = os.path.join(output_dir, 'TEST', 'NORMAL')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all nifti files
nifti_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

# Shuffle the files
random.shuffle(nifti_files)

# Split the files
train_split = int(train_ratio * len(nifti_files))
val_split = int(val_ratio * len(nifti_files))

train_files = nifti_files[:train_split]
val_files = nifti_files[train_split:train_split + val_split]
test_files = nifti_files[train_split + val_split:]

# Copy files to respective directories
for file in train_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files: {len(test_files)}")

