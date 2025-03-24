import os
import random
import shutil
import csv
from pathlib import Path

# Set the seed for reproducibility
random.seed(42)

# Define the paths
data_dir = '../CT_preprocessing/PREPROCESSED_CT_SCANS'
output_dir = 'DATA'
decedentsfile = 'decedents.csv'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create output directories
train_dir = os.path.join(output_dir, 'TRAIN')
val_dir = os.path.join(output_dir, 'VAL')
test_dir = os.path.join(output_dir, 'TEST', 'NORMAL')
strangulation_test_dir = os.path.join(output_dir, 'TEST', 'STRANGULATION')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(strangulation_test_dir, exist_ok=True)

# Get all nifti files
nifti_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

# Read the CSV file and create a dictionary with deidentified_record_number as key
decedents_info = {}
with open(decedentsfile, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        decedents_info[row['deidentified_record_number']] = row

# Filter files based on the conditions
strangulation_files = []
natural_files = []
other_files = []

for file in nifti_files:
    case_number = file.split('-')[1].split('.')[0]
    if case_number in decedents_info:
        info = decedents_info[case_number]
        if info['primary_cause_of_death'] == 'Asphyxia (suffocation, strangulation)' and info['manner_of_death'] == 'Strangled by assailant(s)':
            strangulation_files.append(file)
        elif info['manner_of_death'] == 'Natural':
            natural_files.append(file)
        else:
            other_files.append(file)

# Shuffle the files
random.shuffle(natural_files)

# Split the natural files
train_split = int(train_ratio * len(natural_files))
val_split = int(val_ratio * len(natural_files))

train_files = natural_files[:train_split]
val_files = natural_files[train_split:train_split + val_split]
test_files = natural_files[train_split + val_split:]

# Copy files to respective directories
for file in train_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

for file in strangulation_files:
    new_filename = file.replace('.nii.gz', '_STR.nii.gz')
    shutil.copy(os.path.join(data_dir, file), os.path.join(strangulation_test_dir, new_filename))

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files (Natural): {len(test_files)}")
print(f"Test files (Strangulation): {len(strangulation_files)}")

