import os
import copy
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免绘图报错
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# --------------------------
# 配置参数
# --------------------------
MARKER_FILES = {
    "task0": "data/Malignant_vs_Benign.markers.txt",
    'task1': 'data/LUAD.markers.txt',
    'task3': 'data/MIA_ADC_vs_AIS.markers.txt'  # 仅保留task1和task3
}

# 外部验证集路径配置
EXTERNAL_EV_TRANSCRIPT_PATH = "data/validation_tpm.txt"  # 外部EV表达矩阵
EXTERNAL_EXCEL_PATH = "data/validation_sample_info.xlsx"   # 外部样本信息Excel
EXTERNAL_IMAGE_DIR = "validation_ct_lesions" # 外部CT图片目录
EXTERNAL_RESULT_DIR = "external_result"  # 结果保存目录
os.makedirs(EXTERNAL_RESULT_DIR, exist_ok=True)

# 核心：EV_id前缀配置（
EV_ID_PREFIX = "N"  # 外部测试集表达矩阵中EV_id的前缀（原EV_id=C2E10 → 表达矩阵中=NC2E10）
EV_ID_COLUMN = "EV_id"
DIAGNOSIS_COLUMN = "Pathology.Diagnosis"
LESION_IMAGE_COLUMN = "Lesion_Image_Names"

# 固定参数
FIXED_IMAGE_SIZE = (224, 224)
MAX_CHANNELS = 5
BATCH_SIZE = 8
NUM_WORKERS = 4
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_CLASSES = ['ADC', 'AIS', 'Benign', 'MIA', 'SCC','Malignant']
# 仅保留task1和task3
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

print(f"使用设备: {DEVICE}")
print(f"EV_id前缀配置: {EV_ID_PREFIX} (外部测试集EV_id = {EV_ID_PREFIX} + Excel中的EV_id)")
print(f"仅运行任务: {[TASKS[tid]['name'] for tid in TASKS.keys()]}")

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
        raise FileNotFoundError(f"Marker基因文件未找到: {marker_file_path}")
    
    marker_df = pd.read_csv(marker_file_path, sep='\t')
    marker_geneids = marker_df.iloc[:, 0].astype(str).tolist()
    print(f"从{marker_file_path}加载{len(marker_geneids)}个marker基因")
    return marker_geneids

def load_ev_transcript_data(file_path, marker_geneids=None):
    """加载并处理EV表达数据（适配外部测试集前缀）"""
    print(f"从{file_path}加载EV转录组数据...")
    
    ev_data = pd.read_csv(file_path, sep='\t', index_col=0)
    if 'gene_name' not in ev_data.columns:
        raise ValueError("EV表达数据必须包含'gene_name'列")
    
    # 分离gene_name和表达数据
    gene_names = ev_data['gene_name']
    expression_data = ev_data.drop(columns=['gene_name'])
    
    # 过滤全零基因
    non_zero_mask = (expression_data.sum(axis=1) > 0)
    filtered_expression = expression_data[non_zero_mask]
    filtered_geneids = filtered_expression.index.astype(str)
    print(f"过滤掉{sum(~non_zero_mask)}个全零表达的基因")
    
    # 过滤marker基因
    if marker_geneids is not None:
        marker_mask = filtered_geneids.isin(marker_geneids)
        filtered_expression = filtered_expression[marker_mask]
        filtered_geneids = filtered_geneids[marker_mask]
        print(f"过滤后保留{len(filtered_expression)}个marker基因")
    
    if len(filtered_expression) == 0:
        raise ValueError("过滤后无有效基因")
    
    # 转置（样本为行，基因为列）
    ev_transposed = filtered_expression.T
    ev_transposed.columns = [f"{gene_names[geneid]} ({geneid})" for geneid in filtered_geneids]
    
    print(f"加载完成：{ev_transposed.shape[0]}个样本, {ev_transposed.shape[1]}个基因")
    print(f"样本ID示例（前5个）: {list(ev_transposed.index[:5])}")
    return ev_transposed

def train_marker_rf(marker_features, marker_labels, n_estimators=300, random_state=42):
    """训练RF模型（匹配训练代码）"""
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
    print("Top 10重要marker基因:")
    print(feature_importance.head(10))
    
    return rf, rf.predict_proba(marker_features)

# --------------------------
# 2. 数据集类
# --------------------------
class CT_Marker_Dataset(torch.utils.data.Dataset):
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
        
        required_columns = ['ID', DIAGNOSIS_COLUMN, LESION_IMAGE_COLUMN, EV_ID_COLUMN]
        missing_columns = [col for col in required_columns if col not in self.original_df.columns]
        if missing_columns:
            raise ValueError(f"缺失必要列: {', '.join(missing_columns)}")
        
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
            print(f"当前任务类别: {self.classes}")
        else:
            self.classes = []
            self.num_classes = 0
            self.class_mapping = {}
            print("当前任务无有效数据")

    def _process_data(self, dataframe):
        """处理数据（核心：适配EV_id前缀）"""
        processed_rows = []
        patient_ids = dataframe['ID'].unique()
        
        for patient_id in patient_ids:
            patient_records = self.original_df[self.original_df['ID'] == patient_id]
            if len(patient_records) == 0:
                continue
                
            original_diagnosis = patient_records.iloc[0][DIAGNOSIS_COLUMN]
            ev_id = patient_records.iloc[0][EV_ID_COLUMN]
            
            
            ev_id_str = str(ev_id).strip()
            ev_id_with_prefix = f"{EV_ID_PREFIX}{ev_id_str}" if EV_ID_PREFIX else ev_id_str
            
            
            if pd.isna(ev_id) or ev_id_with_prefix not in self.ev_data.index:
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
                    'EV_id': ev_id_str,  # 原始EV_id
                    'EV_id_with_prefix': ev_id_with_prefix,  # 带前缀的EV_id（匹配表达矩阵）
                    'Original_Diagnosis': original_diagnosis,
                    'Task_Label': task_label,
                    'Lesion_Image_Names': all_images
                })
        
        result_df = pd.DataFrame(processed_rows)
        if len(result_df) > 0:
            print(f"任务标签分布: {result_df['Task_Label'].value_counts().to_dict()}")
            print(f"匹配的样本ID示例（带前缀）: {result_df['EV_id_with_prefix'].head(5).tolist()}")
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
            
        # 使用带前缀的EV_id匹配表达矩阵
        ev_ids = self.processed_data['EV_id_with_prefix'].astype(str)
        marker_features = self.ev_data.loc[ev_ids].values
        
        scaler = StandardScaler()
        marker_features_scaled = scaler.fit_transform(marker_features)
        print(f"Marker基因特征形状: {marker_features_scaled.shape}")
        return marker_features_scaled, scaler

    def _align_rf_probs(self):
        # 使用带前缀的EV_id匹配RF概率
        ev_ids = self.processed_data['EV_id_with_prefix'].astype(str)
        return self.marker_rf_probs.loc[ev_ids].values

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
                print(f"警告：图片{img_path}缺失，使用空白图片")
                channels.append(torch.zeros((1, target_h, target_w)))
                continue
            
            try:
                img = Image.open(img_path).convert('L')
                img = transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.LANCZOS)(img)
                img = self.transform(img) if self.transform else transforms.ToTensor()(img)
                channels.append(img)
            except Exception as e:
                print(f"处理图片{img_name}出错: {e}，使用空白图片")
                channels.append(torch.zeros((1, target_h, target_w)))
        
        while len(channels) < self.max_channels:
            channels.append(torch.zeros((1, target_h, target_w)))
        image_tensor = torch.cat(channels, dim=0)
        
        marker_rf_feature = torch.tensor(self.marker_rf_probs[idx], dtype=torch.float32)
        
        return image_tensor, marker_rf_feature, label

# --------------------------
# 3. 模型定义
# --------------------------
class CT_Marker_FusionClassifier(nn.Module):
    def __init__(self, input_channels=MAX_CHANNELS, num_rf_features=2, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.image_backbone = models.resnet18(pretrained=False)
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

    def predict(self, image, rf_feat):
        output = self.forward(image, rf_feat)
        preds = (output > 0.5).float()
        return preds, output

class TaskEnsembleModel(nn.Module):
    """集成模型类（完全保留你的可运行代码）"""
    def __init__(self, base_model_class, model_config, model_weights_list=None, weights=None):
        super().__init__()
        self.model_config = model_config
        self.models = nn.ModuleList()
        
        # 加载子模型
        if model_weights_list is not None and len(model_weights_list) > 0:
            for weight_dict in model_weights_list:
                model = base_model_class(
                    input_channels=model_config['channels'],
                    num_rf_features=model_config['num_rf_features']
                ).to(DEVICE)
                model.load_state_dict(weight_dict['model_state_dict'], strict=False)
                model.eval()
                self.models.append(model)
        
        # 集成权重
        if weights is None:
            self.weights = torch.ones(len(self.models)) / len(self.models) if len(self.models) > 0 else torch.tensor([])
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
            self.weights = self.weights / self.weights.sum()
        
        # 模型配置
        self.classes = model_config['classes']
        self.image_size = model_config['image_size']
        self.channels = model_config['channels']
        self.num_rf_features = model_config['num_rf_features']
        self.marker_scaler = model_config['marker_scaler']

    def forward(self, image, rf_feat):
        with torch.no_grad():
            if len(self.models) == 0:
                raise ValueError("集成模型中无可用子模型")
            
            outputs = []
            for model in self.models:
                output = model(image, rf_feat)
                outputs.append(output)
            
            outputs_tensor = torch.stack(outputs)
            weighted_outputs = outputs_tensor * self.weights.view(-1, 1, 1)
            final_output = weighted_outputs.sum(dim=0)
            return final_output

    def predict(self, image, rf_feat):
        output = self.forward(image, rf_feat)
        preds = (output > 0.5).float()
        return preds, output



# --------------------------
# 4. 加载集成模型
# --------------------------
def load_ensemble_model(task_name):
    """直接加载fold子模型构建集成，手动构造model_config，不依赖综合模型"""
    safe_task_name = task_name.replace(" ", "_").lower().replace("/", "_or_")
    model_pattern = f'ct_marker_fusion_{safe_task_name}_fold*_round*.pth'
    model_files = glob.glob(model_pattern)
    
    if len(model_files) == 0:
        raise FileNotFoundError(f"未找到任何子模型文件，匹配规则: {model_pattern}\n请检查模型文件是否存在")
    
    # 手动构造model_config，使用全局固定参数
    model_config = {
        "channels": MAX_CHANNELS,
        "num_rf_features": 2,
        "image_size": FIXED_IMAGE_SIZE,
        "classes": None,
        "marker_scaler": None
    }

    # 批量加载所有fold子模型权重
    model_weights_list = []
    for model_file in model_files:
        sub_checkpoint = torch.load(model_file, map_location=DEVICE)
        model_weights_list.append(sub_checkpoint)
    
    # 初始化集成模型
    ensemble_model = TaskEnsembleModel(
        base_model_class=CT_Marker_FusionClassifier,
        model_config=model_config,
        model_weights_list=model_weights_list,
        weights=None
    ).to(DEVICE)
    
    ensemble_model.eval()
    print(f"✅ 成功加载【{task_name}】所有子模型构建集成，共{len(model_files)}个fold子模型")
    return ensemble_model


# --------------------------
# 5. 准备外部测试集数据（完全保留你的可运行代码）
# --------------------------
def prepare_external_test_data(task_id):
    """准备外部测试集数据（适配前缀）"""
    task_info = TASKS[task_id]
    print(f"\n{'='*50} 处理【{task_info['name']}】的外部测试集数据 {'='*50}")
    
    # 1. 加载marker基因和EV表达数据
    marker_geneids = load_marker_genes(MARKER_FILES[task_id])
    ev_data = load_ev_transcript_data(EXTERNAL_EV_TRANSCRIPT_PATH, marker_geneids)
    
    # 2. 加载外部样本信息
    original_df = pd.read_excel(EXTERNAL_EXCEL_PATH)
    ev_id_label = original_df[[EV_ID_COLUMN, DIAGNOSIS_COLUMN]].dropna()
    ev_id_label[EV_ID_COLUMN] = ev_id_label[EV_ID_COLUMN].astype(str).str.strip()
    
    # 3. 筛选有效样本（适配前缀）
    valid_ev_ids = []          # 原始EV_id（如C2E10）
    valid_ev_ids_with_prefix = []  # 带前缀的EV_id（如NC2E10）
    valid_labels = []
    
    for _, row in ev_id_label.iterrows():
        ev_id = row[EV_ID_COLUMN]
        diag = row[DIAGNOSIS_COLUMN]
        task_label = task_info['mapping'].get(diag, None)
        
        # 为原始EV_id添加前缀
        ev_id_with_prefix = f"{EV_ID_PREFIX}{ev_id}" if EV_ID_PREFIX else ev_id
        
        # 检查是否匹配
        if task_label and ev_id_with_prefix in ev_data.index:
            valid_ev_ids.append(ev_id)
            valid_ev_ids_with_prefix.append(ev_id_with_prefix)
            valid_labels.append(task_label)
    
    # 验证有效样本数
    if len(valid_ev_ids) < 1:
        print(f"⚠️  未找到匹配的有效样本")
        print(f"  - Excel中的EV_id（前10个）: {ev_id_label[EV_ID_COLUMN].head(10).tolist()}")
        print(f"  - 带前缀的EV_id（前10个）: {[f'{EV_ID_PREFIX}{id}' for id in ev_id_label[EV_ID_COLUMN].head(10).tolist()]}")
        print(f"  - 表达矩阵中的样本ID（前10个）: {list(ev_data.index[:10])}")
        raise ValueError(f"外部测试集有效样本数为0，请检查EV_id前缀配置或数据格式")
    
    print(f"外部测试集有效样本数: {len(valid_ev_ids)}")
    print(f"匹配的样本ID示例（带前缀）: {valid_ev_ids_with_prefix[:5]}")
    
    # 4. 训练RF模型（匹配训练流程）
    marker_features = ev_data.loc[valid_ev_ids_with_prefix]
    marker_labels_encoded = LabelEncoder().fit_transform(valid_labels)
    
    rf_model, rf_probs = train_marker_rf(marker_features, marker_labels_encoded)
    
    # 5. 构建RF概率DataFrame（索引为带前缀的EV_id）
    rf_probs_df = pd.DataFrame(
        rf_probs,
        index=valid_ev_ids_with_prefix,
        columns=[f'rf_prob_{c}' for c in task_info['classes']]
    )
    
    # 6. 构建Dataset和DataLoader
    val_transform = transforms.Compose([
        transforms.Resize(FIXED_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    patient_df = pd.DataFrame({'ID': original_df['ID'].unique()})
    test_dataset = CT_Marker_Dataset(
        patient_df, original_df, ev_data, rf_probs_df,
        EXTERNAL_IMAGE_DIR, transform=val_transform, task_mapping=task_info['mapping']
    )
    
    if len(test_dataset) == 0:
        raise ValueError("测试集Dataset为空，请检查CT图片路径或样本信息")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    print(f"✅ 外部测试集DataLoader构建完成，样本数: {len(test_dataset)}")
    return test_loader, test_dataset.classes, test_dataset.processed_data

# --------------------------
# 6. 执行外部测试集预测
# --------------------------
def run_external_prediction(task_id):
    """执行外部测试集预测并保存结果"""
    task_info = TASKS[task_id]
    safe_task_name = task_info['name'].replace(" ", "_").lower().replace("/", "_or_")
    
    # 1. 加载模型和数据
    model = load_ensemble_model(task_info['name'])
    test_loader, classes, test_processed_data = prepare_external_test_data(task_id)
    
    # 2. 执行预测
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    all_original_diags = []
    all_ev_ids = []
    
    with torch.no_grad():
        for batch_idx, (images, rf_feats, labels) in enumerate(tqdm(test_loader, desc="外部测试集预测")):
            images = images.to(DEVICE)
            rf_feats = rf_feats.to(DEVICE)
            
            # 模型预测
            preds, probs = model.predict(images, rf_feats)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            
            # 收集样本信息
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + len(images)
            batch_data = test_processed_data.iloc[start_idx:end_idx]
            all_patient_ids.extend(batch_data['ID'].tolist())
            all_original_diags.extend(batch_data['Original_Diagnosis'].tolist())
            all_ev_ids.extend(batch_data['EV_id_with_prefix'].tolist())
    
    # 3. 结果转换
    label_encoder = test_loader.dataset.label_encoder
    pred_labels = label_encoder.inverse_transform(np.array(all_preds).astype(int))
    true_labels = label_encoder.inverse_transform(np.array(all_labels).astype(int))
    
    # 4. 保存预测结果（确保长度一致）
    result_df = pd.DataFrame({
        'Patient_ID': all_patient_ids[:len(true_labels)],
        'EV_id': all_ev_ids[:len(true_labels)],
        'Original_Diagnosis': all_original_diags[:len(true_labels)],
        'True_Label': true_labels,
        'Pred_Label': pred_labels,
        'Pred_Probability': all_probs,
        'Is_Correct': (true_labels == pred_labels)
    })
    result_path = os.path.join(EXTERNAL_RESULT_DIR, f'external_test_{safe_task_name}_predictions.xlsx')
    result_df.to_excel(result_path, index=False)
    print(f"\n✅ 预测结果已保存至: {result_path}")
    
    # 5. 计算评估指标
    print(f"\n{'='*50} 【{task_info['name']}】外部测试集评估结果 {'='*50}")
    acc = np.mean(true_labels == pred_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = f"无法计算: {str(e)}"
    
    print(f"准确率(Accuracy): {acc:.4f}")
    print(f"F1分数(F1-Score): {f1:.4f}")
    print(f"AUC: {auc}")
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, target_names=classes))
    
    # 6. 保存混淆矩阵（数据+可视化）
    cm = confusion_matrix(true_labels, pred_labels)
    # 保存混淆矩阵原始数据
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{c}' for c in classes],
        columns=[f'Pred_{c}' for c in classes]
    )
    cm_data_path = os.path.join(EXTERNAL_RESULT_DIR, f'{safe_task_name}_confusion_matrix_data.xlsx')
    cm_df.to_excel(cm_data_path)
    print(f"混淆矩阵原始数据已保存至: {cm_data_path}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'External Test Set Confusion Matrix - {task_info["name"]}')
    plt.tight_layout()
    cm_path = os.path.join(EXTERNAL_RESULT_DIR, f'external_test_{safe_task_name}_confusion_matrix.pdf')
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵可视化图已保存至: {cm_path}")
    
    # 7. 保存ROC曲线（数据+可视化，二分类任务）
    if len(classes) == 2 and isinstance(auc, float):
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        # 保存ROC原始数据
        roc_df = pd.DataFrame({
            'False_Positive_Rate (FPR)': fpr,
            'True_Positive_Rate (TPR)': tpr,
            'Threshold': thresholds,
            'AUC': [auc] * len(fpr)
        })
        roc_data_path = os.path.join(EXTERNAL_RESULT_DIR, f'{safe_task_name}_roc_data.xlsx')
        roc_df.to_excel(roc_data_path)
        print(f"ROC曲线原始数据已保存至: {roc_data_path}")
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'External Test Set ROC Curve - {task_info["name"]}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(EXTERNAL_RESULT_DIR, f'external_test_{safe_task_name}_roc_curve.pdf')
        plt.savefig(roc_path)
        plt.close()
        print(f"ROC曲线可视化图已保存至: {roc_path}")
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc,
        'result_df': result_df
    }

# --------------------------
# 7. 主函数
# --------------------------
def main():
    print(f"开始处理外部验证集，结果保存至: {EXTERNAL_RESULT_DIR}")
    print(f"仅执行任务: {[TASKS[tid]['name'] for tid in TASKS.keys()]}")
    
    
    tasks_to_predict = ['task0','task1', 'task3']
    all_results = {}
    
    for task_id in tasks_to_predict:
        try:
            print(f"\n{'='*60} 开始处理任务: {TASKS[task_id]['name']} {'='*60}")
            results = run_external_prediction(task_id)
            all_results[task_id] = results
        except Exception as e:
            print(f"❌ 任务【{TASKS[task_id]['name']}】预测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印汇总结果
    print(f"\n{'='*60} 所有任务预测汇总 {'='*60}")
    for task_id, results in all_results.items():
        task_name = TASKS[task_id]['name']
        print(f"- {task_name}: 准确率={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
