import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# --------------------------
# 配置参数（含ROC坐标保存目录）
# --------------------------
MARKER_FILES = {
    "task0": "data/Malignant_vs_Benign.markers.txt",
    'task1': 'data/LUAD.markers.txt',
    'task2': 'data/SCC.markers.txt',
    'task3': 'data/MIA_ADC_vs_AIS.markers.txt'
}

EV_TRANSCRIPT_PATH = "data/training_tpm.txt"
EV_ID_COLUMN = "EV_id"
EXCEL_PATH = "data/training_sample_info.xlsx"
IMAGE_DIR = "training_ct_lesions"
DIAGNOSIS_COLUMN = "Pathology.Diagnosis"
LESION_IMAGE_COLUMN = "Lesion_Image_Names"
ROC_COORD_DIR = "fusion_roc_coordinates"  # ROC坐标保存目录
os.makedirs(ROC_COORD_DIR, exist_ok=True)  # 自动创建目录

SAMPLING_STRATEGY = 'upsample_minority'
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.5
FREEZE_LAYERS = 50
MAX_CHANNELS = 5
NUM_WORKERS = 4
NUM_CV_FOLDS = 5
NUM_SAMPLING_ROUNDS = 10
FIXED_IMAGE_SIZE = (224, 224)
SAMPLING_REPLACEMENT = True

ALL_CLASSES = ['ADC', 'AIS', 'Benign', 'MIA', 'SCC',"Malignant"]
TASKS = {
    'task0': {
        'name': 'Benign vs Malignant',
        'classes': ['Benign', 'Malignant'],
        'mapping': {
            'Benign': 'Benign',
            'AIS': 'Malignant',
            'MIA': 'Malignant',
            'ADC': 'Malignant',
            'SCC': 'Malignant'
        }
    },
    'task1': {
        'name': 'Benign vs LUAD',
        'classes': ['Benign', 'LUAD'],
        'mapping': {
            'Benign': 'Benign',
            'AIS': 'LUAD',
            'MIA': 'LUAD',
            'ADC': 'LUAD',
            'SCC': None
        }
    },
    'task2': {
        'name': 'Benign vs SCC',
        'classes': ['Benign', 'SCC'],
        'mapping': {
            'Benign': 'Benign',
            'SCC': 'SCC',
            'AIS': None,
            'MIA': None,
            'ADC': None
        }
    },
    'task3': {
        'name': 'AIS vs MIA/ADC',
        'classes': ['AIS', 'MIA/ADC'],
        'mapping': {
            'AIS': 'AIS',
            'MIA': 'MIA/ADC',
            'ADC': 'MIA/ADC',
            'Benign': None,
            'SCC': None
        }
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --------------------------
# 1. 工具函数
# --------------------------
def safe_isna(value):
    if isinstance(value, np.ndarray):
        return value.size == 0 or np.any(np.isnan(value))
    if isinstance(value, list):
        return len(value) == 0 or any(safe_isna(item) for item in value)
    return pd.isna(value)


def load_marker_genes(marker_file_path):
    if not os.path.exists(marker_file_path):
        raise FileNotFoundError(f"Marker gene file not found: {marker_file_path}")
    
    marker_df = pd.read_csv(marker_file_path, sep='\t')
    marker_geneids = marker_df.iloc[:, 0].astype(str).tolist()
    print(f"Loaded {len(marker_geneids)} marker genes from {marker_file_path}")
    return marker_geneids


def load_ev_transcript_data(file_path, marker_geneids=None):
    print(f"Loading exosome transcriptome data from {file_path}...")
    
    ev_data = pd.read_csv(file_path, sep='\t', index_col=0)
    if 'gene_name' not in ev_data.columns:
        raise ValueError("EV transcript data must contain 'gene_name' column")
    
    gene_names = ev_data['gene_name']
    expression_data = ev_data.drop(columns=['gene_name'])
    non_zero_mask = (expression_data.sum(axis=1) > 0)
    filtered_expression = expression_data[non_zero_mask]
    filtered_geneids = filtered_expression.index.astype(str)
    print(f"Filtered out {sum(~non_zero_mask)} genes with all zero expression")
    
    if marker_geneids is not None:
        marker_mask = filtered_geneids.isin(marker_geneids)
        filtered_expression = filtered_expression[marker_mask]
        filtered_geneids = filtered_geneids[marker_mask]
        print(f"Filtered to {len(filtered_expression)} marker genes")
    
    if len(filtered_expression) == 0:
        raise ValueError("No valid genes left after filtering")
    
    ev_transposed = filtered_expression.T
    ev_transposed.columns = [f"{gene_names[geneid]} ({geneid})" for geneid in filtered_geneids]
    return ev_transposed


def train_marker_rf(marker_features, marker_labels, n_estimators=300, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(marker_features, marker_labels)
    
    feature_importance = pd.DataFrame({
        'feature': marker_features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Top 10 important marker genes:")
    print(feature_importance.head(10))
    
    return rf, rf.predict_proba(marker_features)


# --------------------------
# 2. 数据集类
# --------------------------
class CT_Marker_Dataset(Dataset):
    def __init__(self, dataframe, original_df, ev_data, marker_rf_probs, 
                 image_dir, transform=None, task_mapping=None):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = FIXED_IMAGE_SIZE
        self.max_channels = MAX_CHANNELS
        self.task_mapping = task_mapping
        self.original_df = original_df
        self.ev_data = ev_data
        self.marker_rf_probs = marker_rf_probs
        
        print(f"Using fixed image size: {self.target_size}")
        
        required_columns = ['ID', DIAGNOSIS_COLUMN, LESION_IMAGE_COLUMN, EV_ID_COLUMN]
        missing_columns = [col for col in required_columns if col not in self.original_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        self.processed_data = self._process_data(dataframe)
        self.marker_features, self.marker_scaler = self._prepare_marker_features()
        self.marker_rf_probs = self._align_rf_probs()
        
        if len(self.processed_data) > 0:
            self.label_encoder = LabelEncoder()
            self.processed_data['Encoded_Label'] = self.label_encoder.fit_transform(
                self.processed_data['Task_Label'])
            self.classes = list(self.label_encoder.classes_)
            self.num_classes = len(self.classes)
            self.class_mapping = dict(zip(self.classes, range(len(self.classes))))
            print(f"Current task classes: {self.classes}")
        else:
            self.classes = []
            self.num_classes = 0
            self.class_mapping = {}
            print("No available data for current task")

    def _process_data(self, dataframe):
        processed_rows = []
        patient_ids = dataframe['ID'].unique()
        
        for patient_id in patient_ids:
            patient_records = self.original_df[self.original_df['ID'] == patient_id]
            if len(patient_records) == 0:
                continue
                
            original_diagnosis = patient_records.iloc[0][DIAGNOSIS_COLUMN]
            ev_id = patient_records.iloc[0][EV_ID_COLUMN]
            
            if pd.isna(ev_id) or str(ev_id) not in self.ev_data.index:
                print(f"Warning: No marker data for patient {patient_id} (EV_id: {ev_id}), skipping")
                continue
                
            if original_diagnosis not in ALL_CLASSES:
                continue
                
            task_label = self.task_mapping.get(original_diagnosis, None) if self.task_mapping else original_diagnosis
            if task_label is None:
                continue
            
            all_images = []
            for _, row in patient_records.iterrows():
                lesion_images = row[LESION_IMAGE_COLUMN]
                if self._is_empty_or_invalid(lesion_images):
                    continue
                
                if isinstance(lesion_images, str):
                    image_names = [img.strip() for img in lesion_images.split(',') if img.strip()]
                elif isinstance(lesion_images, list):
                    image_names = [str(img).strip() for img in lesion_images if not safe_isna(img)]
                else:
                    image_names = [str(lesion_images).strip()] if str(lesion_images).strip() else []
                all_images.extend(image_names)
            
            if all_images:
                processed_rows.append({
                    'ID': patient_id,
                    'EV_id': ev_id,
                    'Original_Diagnosis': original_diagnosis,
                    'Task_Label': task_label,
                    'Lesion_Image_Names': all_images
                })
        
        result_df = pd.DataFrame(processed_rows)
        if len(result_df) > 0:
            print(f"Processed task label distribution: {result_df['Task_Label'].value_counts().to_dict()}")
        return result_df

    def _is_empty_or_invalid(self, value):
        if safe_isna(value):
            return True
        if isinstance(value, (np.ndarray, list)):
            return len(value) == 0 or all(safe_isna(item) or str(item).strip() == '' for item in value)
        return str(value).strip() == ''

    def _prepare_marker_features(self):
        if len(self.processed_data) == 0:
            return None, None
            
        ev_ids = self.processed_data['EV_id'].astype(str)
        marker_features = self.ev_data.loc[ev_ids].values
        
        scaler = StandardScaler()
        marker_features_scaled = scaler.fit_transform(marker_features)
        print(f"Marker gene features shape: {marker_features_scaled.shape}")
        return marker_features_scaled, scaler

    def _align_rf_probs(self):
        ev_ids = self.processed_data['EV_id'].astype(str)
        return self.marker_rf_probs.loc[ev_ids].values

    def get_raw_image(self, patient_idx, lesion_idx=0):
        row = self.processed_data.iloc[patient_idx]
        image_names = row['Lesion_Image_Names']
        if lesion_idx >= len(image_names):
            raise IndexError(f"Lesion index {lesion_idx} out of range")
            
        image_path = os.path.join(self.image_dir, image_names[lesion_idx])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path).convert('L')
        return image, image_names[lesion_idx], row['Original_Diagnosis'], row['Task_Label']

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx]
        image_names = row['Lesion_Image_Names']
        label = row['Encoded_Label']
        
        channels = []
        target_h, target_w = self.target_size
        for i, img_name in enumerate(image_names[:self.max_channels]):
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} missing, using blank")
                channels.append(torch.zeros((1, target_h, target_w)))
                continue
            
            try:
                img = Image.open(img_path).convert('L')
                img = transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.LANCZOS)(img)
                img = self.transform(img) if self.transform else transforms.ToTensor()(img)
                channels.append(img)
            except Exception as e:
                print(f"Error processing {img_name}: {e}, using blank")
                channels.append(torch.zeros((1, target_h, target_w)))
        
        while len(channels) < self.max_channels:
            channels.append(torch.zeros((1, target_h, target_w)))
        image_tensor = torch.cat(channels, dim=0)
        
        marker_rf_feature = torch.tensor(self.marker_rf_probs[idx], dtype=torch.float32)
        
        return image_tensor, marker_rf_feature, label


# --------------------------
# 3. 数据平衡与交叉验证
# --------------------------
def create_stratified_folds(dataset, n_splits=NUM_CV_FOLDS):
    labels = dataset.processed_data['Encoded_Label'].values
    indices = np.arange(len(dataset))
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Need exactly 2 classes, got {len(unique_labels)}")
    
    class0_cnt = np.sum(labels == unique_labels[0])
    class1_cnt = np.sum(labels == unique_labels[1])
    minority_label = unique_labels[0] if class0_cnt <= class1_cnt else unique_labels[1]
    minority_cnt = min(class0_cnt, class1_cnt)
    n_splits = min(n_splits, minority_cnt)
    print(f"Minority class: {minority_label} (count: {minority_cnt}), adjusted folds: {n_splits}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(skf.split(indices, labels))


def balance_classes(train_indices, dataset, strategy=SAMPLING_STRATEGY, replacement=SAMPLING_REPLACEMENT):
    labels = dataset.processed_data['Encoded_Label'].values[train_indices]
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        return train_indices
    
    class0_idx = train_indices[labels == unique_labels[0]]
    class1_idx = train_indices[labels == unique_labels[1]]
    minority_idx, majority_idx = (class0_idx, class1_idx) if len(class0_idx) <= len(class1_idx) else (class1_idx, class0_idx)
    
    if strategy == 'downsample_majority':
        sampled_majority = np.random.choice(majority_idx, size=len(minority_idx), replace=replacement)
        balanced_indices = np.concatenate([minority_idx, sampled_majority])
        np.random.shuffle(balanced_indices)
        return balanced_indices
    elif strategy == 'upsample_minority':
        sampled_minority = np.random.choice(minority_idx, size=len(majority_idx), replace=replacement)
        balanced_indices = np.concatenate([sampled_minority, majority_idx])
        np.random.shuffle(balanced_indices)
        return balanced_indices
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(size=FIXED_IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(FIXED_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform, val_transform


# --------------------------
# 4. 融合模型定义
# --------------------------
class CT_Marker_FusionClassifier(nn.Module):
    def __init__(self, input_channels=MAX_CHANNELS, num_rf_features=2, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.image_backbone = models.resnet18(pretrained=True)
        original_conv1 = self.image_backbone.conv1
        self.image_backbone.conv1 = nn.Conv2d(
            input_channels, original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        with torch.no_grad():
            avg_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            self.image_backbone.conv1.weight = nn.Parameter(avg_weight.repeat(1, input_channels, 1, 1))
        
        for param in list(self.image_backbone.parameters())[:-FREEZE_LAYERS]:
            param.requires_grad = False
        
        self.image_feat_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()
        
        self.rf_branch = nn.Sequential(
            nn.Linear(num_rf_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.image_feat_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, rf_feat):
        img_feat = self.image_backbone(image)
        rf_feat = self.rf_branch(rf_feat)
        fused_feat = torch.cat([img_feat, rf_feat], dim=1)
        output = self.fusion(fused_feat)
        return output


# --------------------------
# 5. 训练与验证
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, classes, round_num, fold_num, task_name):
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    if len(classes) < 2:
        print(f"Warning: Insufficient classes for {task_name}")
        return model, history, [], [], []
    
    best_val_f1 = 0.0
    best_weights = None
    positive_class = classes[1]  # 二分类的正类
    
    for epoch in range(num_epochs):
        print(f'\n{task_name} - Fold {fold_num+1} - Round {round_num+1} - Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        for images, rf_feats, labels in tqdm(train_loader, desc="Train"):
            images = images.to(DEVICE)
            rf_feats = rf_feats.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images, rf_feats)
            preds = (outputs > 0.5).float()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        
        # 计算训练指标
        if len(all_labels) > 0 and len(np.unique(all_labels)) == 2:
            train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pos_label_encoded = np.where(np.array(classes) == positive_class)[0][0]
            train_f1 = f1_score(all_labels, all_preds, pos_label=pos_label_encoded)
            history['train_loss'].append(running_loss / len(all_labels))
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            print(f'Train Loss: {running_loss/len(all_labels):.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}')
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, rf_feats, labels in tqdm(val_loader, desc="Val"):
                images = images.to(DEVICE)
                rf_feats = rf_feats.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)
                
                outputs = model(images, rf_feats)
                preds = (outputs > 0.5).float()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(outputs.cpu().numpy().flatten())
        
        # 计算验证指标
        if len(all_labels) > 0 and len(np.unique(all_labels)) == 2:
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pos_label_encoded = np.where(np.array(classes) == positive_class)[0][0]
            val_f1 = f1_score(all_labels, all_preds, pos_label=pos_label_encoded)
            history['val_loss'].append(running_loss / len(all_labels))
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            print(f'Val Loss: {running_loss/len(all_labels):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())
    
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f'Best Val F1: {best_val_f1:.4f}')
    return model, history, all_preds, all_labels, all_probs


# --------------------------
# 6. 主训练流程（含ROC坐标保存）
# --------------------------
def load_task_data(task_mapping, ev_data, marker_rf_probs, visualize=True):
    original_df = pd.read_excel(EXCEL_PATH)
    print(f"Excel columns: {original_df.columns.tolist()}")
    
    required_cols = ['ID', DIAGNOSIS_COLUMN, LESION_IMAGE_COLUMN, EV_ID_COLUMN]
    missing_cols = [c for c in required_cols if c not in original_df.columns]
    if missing_cols:
        raise ValueError(f"Missing Excel columns: {', '.join(missing_cols)}")
    
    patient_df = pd.DataFrame({'ID': original_df['ID'].unique()})
    dataset = CT_Marker_Dataset(
        patient_df, original_df, ev_data, marker_rf_probs,
        IMAGE_DIR, task_mapping=task_mapping
    )
    
    if visualize and len(dataset) > 0:
        fig, axes = plt.subplots(1, min(5, len(dataset)), figsize=(15, 4))
        axes = [axes] if min(5, len(dataset)) == 1 else axes
        for i in range(min(5, len(dataset))):
            try:
                img, img_name, diag, task_label = dataset.get_raw_image(i)
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"Sample {i+1}\nDiag: {diag}\nLabel: {task_label}")
                axes[i].axis('off')
            except Exception as e:
                axes[i].set_title(f"Sample {i+1} Load Fail")
                axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('task_marker_images_visualization.pdf')
        plt.close()
    
    if len(dataset) == 0:
        print("No valid data for task")
        return None, original_df, []
    return dataset, original_df, dataset.classes


def train_with_sampling_cv(full_dataset, original_df, classes, task_name):
    if len(classes) < 2:
        print(f"Error: Insufficient classes for {task_name}")
        return []
    
    num_rf_features = full_dataset.marker_rf_probs.shape[1]
    print(f"Marker RF feature dimension: {num_rf_features}")
    
    try:
        folds = create_stratified_folds(full_dataset)
        print(f"Created {len(folds)} folds")
    except ValueError as e:
        print(f"Fold creation failed: {e}")
        return []
    
    train_transform, val_transform = get_transforms()
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*40} Fold {fold+1}/{len(folds)} {'='*40}")
        
        val_dataset = copy.deepcopy(full_dataset)
        val_dataset.transform = val_transform
        val_loader = DataLoader(
            Subset(val_dataset, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        fold_results = {'models': [], 'histories': [], 'all_preds': [], 'all_labels': [], 'all_probs': []}
        
        for round_idx in range(NUM_SAMPLING_ROUNDS):
            print(f"\n{'='*30} Round {round_idx+1}/{NUM_SAMPLING_ROUNDS} {'='*30}")
            
            balanced_train_idx = balance_classes(train_idx, full_dataset)
            train_dataset = copy.deepcopy(full_dataset)
            train_dataset.transform = train_transform
            train_loader = DataLoader(
                Subset(train_dataset, balanced_train_idx),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
            
            model = CT_Marker_FusionClassifier(
                input_channels=MAX_CHANNELS,
                num_rf_features=num_rf_features
            ).to(DEVICE)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
            
            model, history, preds, labels, probs = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                EPOCHS, classes, round_idx, fold, task_name
            )
            
            fold_results['models'].append(model)
            fold_results['histories'].append(history)
            fold_results['all_preds'].extend(preds)
            fold_results['all_labels'].extend(labels)
            fold_results['all_probs'].extend(probs)
            
            safe_task_name = task_name.replace(" ", "_").lower().replace("/", "_or_")
            model_path = f'ct_marker_fusion_{safe_task_name}_fold{fold+1}_round{round_idx+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'image_size': FIXED_IMAGE_SIZE,
                'channels': MAX_CHANNELS,
                'num_rf_features': num_rf_features,
                'marker_scaler': full_dataset.marker_scaler
            }, model_path)
            print(f"Model saved to {model_path}")
        
        print(f"\nFold {fold+1} Overall Results:")
        if len(fold_results['all_labels']) > 0:
            print(classification_report(
                fold_results['all_labels'], fold_results['all_preds'], target_names=classes
            ))
            try:
                auc = roc_auc_score(fold_results['all_labels'], fold_results['all_probs'])
                print(f"Fold {fold+1} AUC: {auc:.4f}")
            except:
                print("Fold AUC calculation failed")
        all_fold_results.append(fold_results)
    
    print(f"\n{'='*50} {task_name} Overall Results {'='*50}")
    all_preds = []
    all_labels = []
    all_probs = []
    for fold in all_fold_results:
        all_preds.extend(fold['all_preds'])
        all_labels.extend(fold['all_labels'])
        all_probs.extend(fold['all_probs'])
    
    if len(all_labels) > 0:
        print(classification_report(all_labels, all_preds, target_names=classes))
        
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{task_name} Confusion Matrix')
        safe_task_name = task_name.replace(" ", "_").lower().replace("/", "_or_")
        plt.savefig(f'{safe_task_name}_marker_fusion_cm.pdf')
        plt.close()
        
        # 核心修改：计算并保存ROC坐标（FPR/TPR/阈值/AUC）
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            auc = roc_auc_score(all_labels, all_probs)
            
            # 保存ROC坐标到CSV
            roc_coord_path = os.path.join(ROC_COORD_DIR, f"{safe_task_name}_fusion_roc_coords.csv")
            roc_coord_df = pd.DataFrame({
                "fpr": fpr,          # 假阳性率（X轴）
                "tpr": tpr,          # 真阳性率（Y轴）
                "threshold": thresholds,  # 分类阈值
                "auc": [auc] * len(fpr)   # AUC值
            })
            roc_coord_df.to_csv(roc_coord_path, index=False)
            print(f"ROC坐标已保存至: {roc_coord_path}")
            
            # 绘制ROC曲线PDF
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'{task_name} ROC Curve')
            plt.legend()
            plt.savefig(f'{safe_task_name}_marker_fusion_roc.pdf')
            plt.close()
        except Exception as e:
            print(f"ROC处理失败: {e}")
    return all_fold_results


# --------------------------
# 主函数
# --------------------------
def main():
    print(f"Sampling strategy: {SAMPLING_STRATEGY} | Rounds: {NUM_SAMPLING_ROUNDS}")
    print(f"Image channels: {MAX_CHANNELS} | Size: {FIXED_IMAGE_SIZE}")
    
    for task_id, task_info in TASKS.items():
        print(f"\n{'='*60} Task: {task_info['name']} {'='*60}")
        
        marker_file = MARKER_FILES.get(task_id)
        if not marker_file:
            print(f"No marker file for {task_id}, skipping")
            continue
        try:
            marker_geneids = load_marker_genes(marker_file)
        except FileNotFoundError as e:
            print(f"Marker file error: {e}, skipping task")
            continue
        
        try:
            ev_data = load_ev_transcript_data(EV_TRANSCRIPT_PATH, marker_geneids)
        except Exception as e:
            print(f"EV data error: {e}, skipping task")
            continue
        
        original_df = pd.read_excel(EXCEL_PATH)
        ev_id_label = original_df[[EV_ID_COLUMN, DIAGNOSIS_COLUMN]].dropna()
        ev_id_label[EV_ID_COLUMN] = ev_id_label[EV_ID_COLUMN].astype(str)
        
        valid_ev_ids = []
        valid_labels = []
        for _, row in ev_id_label.iterrows():
            ev_id = str(row[EV_ID_COLUMN])
            diag = row[DIAGNOSIS_COLUMN]
            task_label = task_info['mapping'].get(diag, None)
            if task_label and ev_id in ev_data.index:
                valid_ev_ids.append(ev_id)
                valid_labels.append(task_label)
        
        if len(valid_ev_ids) < 2:
            print(f"Insufficient samples ({len(valid_ev_ids)}) for Marker RF training, skipping")
            continue
        
        marker_features = ev_data.loc[valid_ev_ids]
        marker_labels = LabelEncoder().fit_transform(valid_labels)
        try:
            rf_model, rf_probs = train_marker_rf(marker_features, marker_labels)
        except Exception as e:
            print(f"RF training error: {e}, skipping task")
            continue
        
        rf_probs_df = pd.DataFrame(
            rf_probs,
            index=valid_ev_ids,
            columns=[f'rf_prob_{c}' for c in task_info['classes']]
        )
        
        full_dataset, _, classes = load_task_data(
            task_info['mapping'], ev_data, rf_probs_df,
            visualize=(task_id == 'task1')
        )
        if full_dataset is None or len(classes) < 2:
            continue
        
        train_with_sampling_cv(full_dataset, original_df, classes, task_info['name'])
        print(f"Task {task_info['name']} completed")
        
        

# --------------------------
# 7. 模型综合（融合）与保存
# --------------------------
class TaskEnsembleModel(nn.Module):
    """任务级别的集成模型，用于综合多个fold/round的模型"""
    def __init__(self, base_model_class, model_config, model_weights_list, weights=None):
        super().__init__()
        self.model_config = model_config
        self.models = nn.ModuleList()
        
        # 加载所有子模型
        for weight_dict in model_weights_list:
            model = base_model_class(
                input_channels=model_config['channels'],
                num_rf_features=model_config['num_rf_features']
            ).to(DEVICE)
            model.load_state_dict(weight_dict['model_state_dict'])
            model.eval()  # 设置为评估模式
            self.models.append(model)
        
        # 集成权重（None表示等权重）
        if weights is None:
            self.weights = torch.ones(len(self.models)) / len(self.models)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()  # 归一化
        
        # 保存模型配置信息
        self.classes = model_config['classes']
        self.image_size = model_config['image_size']
        self.channels = model_config['channels']
        self.num_rf_features = model_config['num_rf_features']
        self.marker_scaler = model_config['marker_scaler']

    def forward(self, image, rf_feat):
        """前向传播：集成所有模型的预测结果"""
        with torch.no_grad():
            outputs = []
            for model in self.models:
                output = model(image, rf_feat)
                outputs.append(output)
            
            # 加权平均
            outputs_tensor = torch.stack(outputs)
            weighted_outputs = outputs_tensor * self.weights.view(-1, 1, 1)
            final_output = weighted_outputs.sum(dim=0)
            
            return final_output

    def predict(self, image, rf_feat):
        """预测接口：返回类别和概率"""
        output = self.forward(image, rf_feat)
        preds = (output > 0.5).float()
        return preds, output

    def save_ensemble_model(self, task_name):
        """保存集成模型到文件"""
        safe_task_name = task_name.replace(" ", "_").lower().replace("/", "_or_")
        save_path = f'final_{safe_task_name}_ensemble_model.pth'
        
        # 保存完整的集成模型
        torch.save({
            'model_state_dict': self.state_dict(),
            'ensemble_weights': self.weights.cpu().numpy(),
            'classes': self.classes,
            'image_size': self.image_size,
            'channels': self.channels,
            'num_rf_features': self.num_rf_features,
            'marker_scaler': self.marker_scaler,
            'model_config': self.model_config
        }, save_path)
        
        print(f"✅ 综合模型已保存至: {save_path}")
        return save_path


def collect_task_models(task_name):
    """收集指定任务的所有训练好的模型文件"""
    safe_task_name = task_name.replace(" ", "_").lower().replace("/", "_or_")
    model_pattern = f'ct_marker_fusion_{safe_task_name}_fold*_round*.pth'
    
    # 查找所有匹配的模型文件
    import glob
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"⚠️ 未找到{task_name}的模型文件")
        return []
    
    print(f"�� 找到{task_name}的{len(model_files)}个模型文件")
    
    # 加载所有模型权重
    model_weights_list = []
    model_config = None
    
    for model_file in model_files:
        checkpoint = torch.load(model_file, map_location=DEVICE)
        model_weights_list.append(checkpoint)
        
        # 只保存一次配置信息
        if model_config is None:
            model_config = {
                'channels': checkpoint['channels'],
                'num_rf_features': checkpoint['num_rf_features'],
                'classes': checkpoint['classes'],
                'image_size': checkpoint['image_size'],
                'marker_scaler': checkpoint['marker_scaler']
            }
    
    return model_weights_list, model_config


def create_ensemble_model_for_task(task_id, task_info):
    """为指定任务创建集成模型"""
    print(f"\n{'='*60} 开始综合 {task_info['name']} 的模型 {'='*60}")
    
    # 收集该任务的所有模型
    model_weights_list, model_config = collect_task_models(task_info['name'])
    if not model_weights_list or model_config is None:
        return None
    
    # 创建集成模型（等权重平均）
    ensemble_model = TaskEnsembleModel(
        base_model_class=CT_Marker_FusionClassifier,
        model_config=model_config,
        model_weights_list=model_weights_list,
        weights=None  # None表示等权重，也可以传入自定义权重
    ).to(DEVICE)
    
    # 保存最终集成模型
    save_path = ensemble_model.save_ensemble_model(task_info['name'])
    
    return ensemble_model, save_path


def load_ensemble_model(task_name):
    """加载保存的集成模型，用于后续外部验证"""
    safe_task_name = task_name.replace(" ", "_").lower().replace("/", "_or_")
    model_path = f'final_{safe_task_name}_ensemble_model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"集成模型文件不存在: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 重建模型
    model_config = checkpoint['model_config']
    ensemble_model = TaskEnsembleModel(
        base_model_class=CT_Marker_FusionClassifier,
        model_config=model_config,
        model_weights_list=[],  # 权重已包含在state_dict中
        weights=checkpoint['ensemble_weights']
    ).to(DEVICE)
    
    ensemble_model.load_state_dict(checkpoint['model_state_dict'])
    ensemble_model.eval()
    
    print(f"✅ 成功加载{task_name}的集成模型")
    return ensemble_model


# --------------------------
# 8. 新增主函数：模型综合入口
# --------------------------
def ensemble_main():
    """模型综合的主函数"""
    print("\n" + "="*80)
    print("开始综合所有任务的模型")
    print("="*80 + "\n")
    
    # 为每个任务创建并保存集成模型
    ensemble_models = {}
    for task_id, task_info in TASKS.items():
        model, save_path = create_ensemble_model_for_task(task_id, task_info)
        if model:
            ensemble_models[task_id] = {
                'model': model,
                'save_path': save_path,
                'task_name': task_info['name']
            }
    
    print("\n" + "="*80)
    print("模型综合完成！生成的最终模型列表：")
    print("="*80)
    for task_id, info in ensemble_models.items():
        print(f"- {info['task_name']}: {info['save_path']}")
    
    return ensemble_models


# --------------------------
# 9. 外部验证集测试示例函数
# --------------------------
def test_with_external_data(ensemble_model, external_data_loader):
    """
    使用集成模型测试外部验证集
    参数：
        ensemble_model: 加载的集成模型
        external_data_loader: 外部验证集的DataLoader（格式需与训练数据一致）
    """
    ensemble_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, rf_feats, labels in tqdm(external_data_loader, desc="外部验证集测试"):
            images = images.to(DEVICE)
            rf_feats = rf_feats.to(DEVICE)
            
            # 使用集成模型预测
            preds, probs = ensemble_model.predict(images, rf_feats)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    # 计算评估指标
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n外部验证集评估结果：")
    print(f"- 准确率(Accuracy): {acc:.4f}")
    print(f"- F1分数: {f1:.4f}")
    print(f"- AUC: {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ensemble_model.classes, 
                yticklabels=ensemble_model.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('External Validation Confusion Matrix')
    plt.savefig('external_validation_cm.pdf')
    plt.close()
    
    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


# 修改原有的main函数，在训练完成后自动执行模型综合
if __name__ == "__main__":
    # 1. 先执行原有训练流程
    main()
    
    # 2. 训练完成后，自动综合所有任务的模型
    #ensemble_main()
    
    # 示例：加载综合模型（后续外部验证时使用）
    # task1_model = load_ensemble_model("Benign vs LUAD")
    # task2_model = load_ensemble_model("Benign vs SCC")
    # task3_model = load_ensemble_model("AIS vs MIA/ADC")
