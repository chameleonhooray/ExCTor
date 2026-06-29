# ExCTor
ExCTor is a predictive tool for non-small cell lung cancer (NSCLC) pathological status, based on exosomal RNA sequencing and low-dose CT (LDCT) imaging. It integrates liquid biopsy molecular features and radiomic characteristics to achieve non-invasive, accurate identification and prediction of NSCLC pathological subtypes and malignancy grades.

Install all required packages via pip:
pip install numpy pandas torch torchvision scikit-learn matplotlib seaborn pillow tqdm

Put all marker files and sample_info files into the data folder.
Download Training_tpm.txt and Validation_tpm.txt of series GSE336118 from the GEO database, then move the two files into the data folder.
Manually create the training_ct_lesions directory to store CT lesion images.
Manually create the validation_ct_lesions directory to store CT lesion images.

Adjust the following global variables in Python scripts to customize file and folder paths:

# Marker gene files for each classification task
MARKER_FILES = {
    "task0": "data/Malignant_vs_Benign.markers.txt",
    'task1': 'data/LUAD.markers.txt',
    'task2': 'data/SCC.markers.txt',
    'task3': 'data/MIA_ADC_vs_AIS.markers.txt'
}

# Transcriptome TPM matrix for training set
EV_TRANSCRIPT_PATH = "data/training_tpm.txt"
# Column name of EV sample ID in expression matrix
EV_ID_COLUMN = "EV_id"
# Excel file containing clinical sample information
EXCEL_PATH = "data/training_sample_info.xlsx"
# Root directory of CT lesion images
IMAGE_DIR = "training_ct_lesions"
# Column name of pathological diagnosis labels
DIAGNOSIS_COLUMN = "Pathology.Diagnosis"
# Column name matching lesion image filenames
LESION_IMAGE_COLUMN = "Lesion_Image_Names"
# Directory to save ROC curve coordinate data
ROC_COORD_DIR = "fusion_roc_coordinates"
# Auto-generate target directory if it does not exist
os.makedirs(ROC_COORD_DIR, exist_ok=True)

First train the model using model_training.py, then perform validation with model_validation.py.
