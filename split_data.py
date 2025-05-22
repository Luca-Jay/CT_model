import os
import random
import shutil
import csv
from pathlib import Path
import pandas as pd

# Set the seed for reproducibility
random.seed(42)

# Define the paths
data_dir = '/workspace/project-data/PREPROCESSED_CT_SCANS/TIGHT'
output_dir = '/workspace/project-data/CT_model/DATA/TIGHT_ALL_HANGING'
decedentsfile = '/workspace/project-data/CT_model/decedents_all.xlsx'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create output directories
train_dir = os.path.join(output_dir, 'TRAIN')
val_dir = os.path.join(output_dir, 'VAL')
test_dir = os.path.join(output_dir, 'TEST', 'NORMAL')
strangulation_test_dir = os.path.join(output_dir, 'TEST', 'STRANGULATION')
asphyxia_test_dir = os.path.join(output_dir, 'TEST', 'ASPHYXIA')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(strangulation_test_dir, exist_ok=True)
os.makedirs(asphyxia_test_dir, exist_ok=True)

# Get all nifti files
nifti_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

# # Read the CSV file and create a dictionary with deidentified_record_number as key
# decedents_info = {}
# with open(decedentsfile, mode='r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         decedents_info[row['deidentified_record_number']] = row


# read by default 1st sheet of an excel file
decedents = pd.read_excel(decedentsfile)

decedents = decedents.set_index('deidentified_record_number', verify_integrity=True)
# Filter files based on the conditions
strangulation_files = []
suffocation_files = []
strangulation_files = []
asphyxia_files = []
natural_files = []
other_files = []

for file in nifti_files:
    # extract and convert case number
    case_number = int(file.split('-')[1].split('.')[0])
    
    # 3) Check if we have metadata for this case
    if case_number not in decedents.index:
        other_files.append(file)
        continue

    # 4) Pull out the one Series for this case
    info = decedents.loc[case_number]
    cause  = info['primary_cause_of_death']
    manner = info['manner_of_death']

    # 5) Now you can do normal scalar comparisons
    if (cause == 'Asphyxia (suffocation, strangulation)'
        and manner == 'Strangled by assailant(s)'):
        strangulation_files.append(file)
    elif cause == 'Asphyxia (suffocation, strangulation)':
        asphyxia_files.append(file)
    elif manner == 'Natural':
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
    if not(os.path.exists(os.path.join(train_dir, file))):
        shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))

for file in val_files:
    if not(os.path.exists(os.path.join(val_dir, file))):
        shutil.copy(os.path.join(data_dir, file), os.path.join(val_dir, file))

for file in test_files:
    if not(os.path.exists(os.path.join(test_dir, file))):
        shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

for file in strangulation_files:
    new_filename = file.replace('.nii.gz', '_STR.nii.gz')
    if not(os.path.exists(os.path.join(strangulation_test_dir, new_filename))):
        shutil.copy(os.path.join(data_dir, file), os.path.join(strangulation_test_dir, new_filename))

for file in asphyxia_files:
    new_filename = file.replace('.nii.gz', '_ASP.nii.gz')
    if not(os.path.exists(os.path.join(asphyxia_test_dir, new_filename))):
        shutil.copy(os.path.join(data_dir, file), os.path.join(asphyxia_test_dir, new_filename))

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files (Natural): {len(test_files)}")
print(f"Test files (Strangulation): {len(strangulation_files)}")
print(f"Test files (Asphyxia): {len(asphyxia_files)}")

