# Imports
from halo_etl.Loading.loaders import PatientLoader, CombineRules
from halo_etl.Loading.load_patients import load_patients
from halo_etl.Loading.patient import Patient, ts_dtype
from halo_etl.Loading.feature_frame import FeatureFrame
from halo_etl.Visualisation.plot import plot, plot_sort_by_num_ts_datapoints, plot_sort_by_num_ts_days
from halo_etl.Transformation.transform import *
from halo_etl.Loading.dataset import Dataset

from functools import partial
import dotenv
import os
import datetime
import torch.nn as nn
import torch
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from torchinfo import summary
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import gc
import copy
import pandas as pd
import random
from scipy.stats import t, sem;
import pickle

# Data filtering
print(dotenv.load_dotenv(r"../../.credentials"))
patient_loader = PatientLoader(
    db_host = os.getenv("DB_HOST"),
    db_username = os.getenv("DB_USERNAME"),
    db_password = os.getenv("DB_PASSWORD"),
)

# remove patients that did not give consent (completely empty record)
patient_loader.apply_patient_filter(
    filter_rule=["389-004-634", "888-129-844", "685-235-185"],
    combine_rule=CombineRules.subtract
)

# Patients that were not allocated with a device
patient_loader.apply_patient_filter(
    filter_rule=["U-1", "U-3", "U-4"],
    combine_rule=CombineRules.subtract
)

# Patients that were not followed up for clinical events
patient_loader.apply_patient_filter(
    filter_rule=["356-937-696", "116-659-625", "421-483-164", "065-993-784"],
    combine_rule=CombineRules.subtract
)


# Patients had no treatment received in observation
patient_loader.apply_patient_filter(
    filter_rule=["814-540-712", "339-311-229", "384-185-791", "625-145-287", "455-919-686"],
    combine_rule=CombineRules.subtract
)

# Patients provide any RPM data for less than 3 days (inc. survey symptom wearable)
patient_loader.apply_patient_filter(
    filter_rule=["462-950-707", "192-608-637", "012-131-721", "701-526-691", "685-235-184", "671-160-910", "646-050-652", "539-744-315", "823-893-878", "741-311-153", "196-608-637"],
    combine_rule=CombineRules.subtract
)

# Patient involved in a car accident during the study (data anomaly)
patient_loader.apply_patient_filter(
    filter_rule=["822-987-952"],
    combine_rule=CombineRules.subtract
)

# Remove Stem Cell Patients (due to distinctive treatment pattern)
patient_loader.apply_patient_filter(
    filter_rule="SELECT DISTINCT id FROM chemo WHERE treatment='Stem cell'",
    combine_rule=CombineRules.subtract
)
patient_loader.apply_patient_filter(
    filter_rule="SELECT DISTINCT id FROM patient_labels WHERE treatment_intended='Stem cell'",
    combine_rule=CombineRules.subtract
)

patients = load_patients(patient_loader)
print("patient left", len(patients))


# Data extraction
intended_features = [
    Patient.feature_age,
    Patient.feature_gender,
    Patient.feature_bmi,

    #=====

    partial(Patient.ts_hr, interval_minutes=60*24, aggregation='max', device_name_filter='HALO-X', clip_range=(40, 200)),
    partial(Patient.ts_steps, interval_minutes=60*24, device_name_filter='HALO-X', clip_range=(0, 600)),
    Patient.ts_daily_wearing_percentage,

    partial(Patient.ts_qor, question_number=[13]), # vomiting
    partial(Patient.ts_qor, question_number=[8]), # return to work
    partial(Patient.ts_qor, question_number=[12]), # sever pain
    partial(Patient.ts_qor, question_number=[4]), # good sleep

    #=====
    
    Patient.ts_wellness_checkin,
    Patient.ts_adverse_events_severity,

    #=====

    partial(Patient.ts_wearable_wearing_gaps, rpm_features=[partial(Patient.ts_hr, aggregation='avg', device_name_filter='HALO-X'), partial(Patient.ts_steps, device_name_filter='HALO-X'), partial(Patient.ts_qor, question_number=list(range(1, 16)))], spacing_thershold=datetime.timedelta(days=1)),

    #=====
    
    partial(Patient.ts_chemo, treatment_type='Chemotherapy'),
    partial(Patient.ts_chemo, treatment_type='Hormone therapy'),
    partial(Patient.ts_chemo, treatment_type='Immunotherapy'),
    partial(Patient.ts_chemo, treatment_type='Mixed therapy'),
    partial(Patient.ts_chemo, treatment_type='Other therapy'),

    #=====

    partial(Patient.ts_readmission, readmission_type='readmission', due_to_deterioration=True),
    partial(Patient.ts_readmission, readmission_type='gp', due_to_deterioration=True),
    partial(Patient.ts_readmission, readmission_type='ae', due_to_deterioration=True),
    Patient.ts_death,
    partial(Patient.ts_treatment_change, treatment_related=True),
]

feature_frame = FeatureFrame(patients, intended_features)
feature_frame.collect_features()


# STraTS-inspired architecture (adapted for type+value tokenization)
# Based on: "Self-supervised transformer for sparse and irregularly sampled multivariate clinical time-series"
# Paper: https://arxiv.org/pdf/2107.14293.pdf
# GitHub: https://github.com/sindhura97/STraTS

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CVE(nn.Module):
    """Continuous Value Embedding - embeds scalar values into hidden dimension"""
    def __init__(self, hidden_dim):
        super().__init__()
        int_dim = int(np.sqrt(hidden_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hidden_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: bsz, max_len
        x = x.unsqueeze(-1)  # bsz, max_len, 1
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # bsz, max_len, int_dim
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # bsz, max_len, hid_dim
        return x


class STraTSTransformer(nn.Module):
    """Multi-head self-attention transformer layers"""
    def __init__(self, num_layers, hidden_dim, num_heads, dropout, attention_dropout):
        super().__init__()
        self.N = num_layers
        self.d = hidden_dim
        self.dff = self.d * 2
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.h = num_heads
        self.dk = self.d // self.h
        self.all_head_size = self.dk * self.h

        # Multi-head attention parameters (one set per layer)
        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.all_head_size, self.d)), requires_grad=True)
        
        # Feed-forward parameters
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)

    def init_proj(self, shape, gain=1):
        """Xavier-like initialization"""
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        x = x * 2 * scale - scale
        return x

    def forward(self, x, mask):
        # x: bsz, max_len, d
        # mask: bsz, max_len
        bsz, max_len, _ = x.size()
        
        # Create attention mask (bsz, 1, max_len, max_len)
        mask_2d = mask[:, :, None] * mask[:, None, :]  # bsz, max_len, max_len
        mask_2d = (1 - mask_2d)[:, None, :, :] * torch.finfo(x.dtype).min
        
        for i in range(self.N):
            # Multi-head attention
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i])  # bsz, h, max_len, dk
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i])  # bsz, h, max_len, dk
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i])  # bsz, h, max_len, dk
            
            # Compute attention scores
            A = torch.einsum('bhle,bhke->bhlk', q, k)  # bsz, h, max_len, max_len
            
            # Apply mask and optional attention dropout
            layer_mask = mask_2d
            if self.training and self.attention_dropout > 0:
                dropout_mask = (torch.rand_like(A) < self.attention_dropout).float() * torch.finfo(x.dtype).min
                layer_mask = mask_2d + dropout_mask
            
            A = A + layer_mask
            A = torch.softmax(A, dim=-1)
            
            # Apply attention to values
            v = torch.einsum('bhkl,bhle->bkhe', A, v)  # bsz, max_len, h, dk
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            
            # Residual connection (without layer norm, as per STraTS improvement)
            x = (all_head_op + x) / 2
            
            # Feed-forward network
            ffn_op = torch.matmul(x, self.W1[i]) + self.b1[i]
            ffn_op = F.gelu(ffn_op)  # GELU instead of ReLU (STraTS improvement)
            ffn_op = torch.matmul(ffn_op, self.W2[i]) + self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            
            # Residual connection
            x = (ffn_op + x) / 2
        
        return x


class FusionAttention(nn.Module):
    """Attention-based pooling to aggregate sequence into single vector"""
    def __init__(self, hidden_dim):
        super().__init__()
        int_dim = hidden_dim
        self.W = nn.Parameter(torch.empty(hidden_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask, return_attention=False):
        # x: bsz, max_len, hid_dim
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # bsz, max_len, int_dim
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # bsz, max_len
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz, max_len
        
        if return_attention:
            return att
        return att


class StandardFeatureProcessor(nn.Module):
    def __init__(self, num_std_features, hidden_dim):
        super().__init__()
        if num_std_features > 0:
            self.std_processor = nn.Sequential(
                nn.Linear(num_std_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            self.std_processor = None
        self.num_std_features = num_std_features

    def forward(self, std_features):
        if self.std_processor:
            return self.std_processor(std_features)
        batch_size = std_features.shape[0] if std_features.numel() > 0 else 1
        return torch.zeros(batch_size, 64, device=std_features.device)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, std_feature_dim=0):
        super().__init__()
        input_dim = hidden_dim + std_feature_dim
        self.ffnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, ts_features, std_features=None):
        x = ts_features if std_features is None else torch.cat([ts_features, std_features], dim=1)
        x = self.ffnn(x)
        return self.output_head(x)


class Model(nn.Module):
    """STraTS-inspired model adapted for type+value tokenization
    
    Key differences from original STraTS:
    - No explicit time embedding (using your existing type+value approach)
    - Adapted to work with your Dataset class
    - Simplified to fit your pipeline
    
    Architecture:
    1. CVE for continuous values + Type embedding
    2. Multi-layer transformer with self-attention
    3. Fusion attention for sequence pooling
    4. Classification head with optional standard features
    """
    def __init__(self, num_types, hidden_dim, num_std_features=0, 
                 num_layers=1, num_heads=2, dropout=0.2, attention_dropout=0.2):
        super().__init__()
        
        # Embedding layers
        self.cve_value = CVE(hidden_dim)
        self.type_embed = nn.Embedding(num_types + 1, hidden_dim, padding_idx=0)
        
        # Transformer encoder
        self.transformer = STraTSTransformer(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        # Fusion attention for pooling
        self.fusion_att = FusionAttention(hidden_dim)
        
        # Standard feature processor
        self.std_processor = StandardFeatureProcessor(num_std_features, hidden_dim // 2)
        
        # Classification head
        self.head = ClassificationHead(hidden_dim, std_feature_dim=hidden_dim // 2)
        
        self.dropout = dropout
        self.num_std_features = num_std_features

    def forward(self, value, type, observation_mask, std_input=None, return_attention=False):
        # Embed value and type
        value_embed = self.cve_value(value)  # bsz, max_len, hidden_dim
        type_embed = self.type_embed(type)   # bsz, max_len, hidden_dim
        
        # Apply mask to embeddings
        mask_expanded = observation_mask.unsqueeze(-1).float()
        value_embed = value_embed * mask_expanded
        type_embed = type_embed * mask_expanded
        
        # Combine embeddings (simple addition as in STraTS)
        doublet_embed = value_embed + type_embed
        doublet_embed = F.dropout(doublet_embed, self.dropout, self.training)
        
        # Pass through transformer
        contextual_embed = self.transformer(doublet_embed, observation_mask)
        
        # Fusion attention pooling
        attention_weights = self.fusion_att(contextual_embed, observation_mask, return_attention=True)
        ts_emb = (contextual_embed * attention_weights[:, :, None]).sum(dim=1)
        
        # Process standard features if available
        if self.num_std_features > 0 and std_input is not None:
            std_features_embed = self.std_processor(std_input)
            output = self.head(ts_emb, std_features_embed)
        else:
            output = self.head(ts_emb)

        if return_attention:
            return output, attention_weights
        return output


hidden_dim = 32
epochs = 80
batch_size = 128
learning_rate = 0.0005
num_layers = 2
num_heads = 4
dropout = 0.1
attention_dropout = 0.1
loss_function = nn.BCEWithLogitsLoss()


def evaluate_model(model, dataset, device, test_sample_size):
    with torch.no_grad():
        data = dataset.test(batch_size=test_sample_size)
        inputs = {key: value.to(device) for key, value in data['input'].items()}
        std_inputs = data['std_input'].to(device)
        targets = data['output'].to(device)

        predictions = model(**inputs, std_input=std_inputs)
        loss = loss_function(predictions, targets.float())

        # Compute AUC
        probs = torch.sigmoid(predictions).cpu().numpy().flatten()  # Convert logits to probabilities
        labels = targets.cpu().numpy().flatten()
        labels = np.where(labels > 0.5, 1, 0)  # Convert to binary labels
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')

        inputs = None  # Release input tensors
        clear_memory()

    return loss.item(), auc, probs, labels


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=5, min_delta=0.0, mode='min', warmup=0):
        """
        Args:
            patience: Number of evaluation steps to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for AUC (higher is better)
            warmup: Number of initial evaluation steps to ignore before monitoring (useful to skip unstable early training)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.warmup = warmup
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
        self.eval_count = 0
        
    def __call__(self, current_value, epoch):
        self.eval_count += 1
        
        if self.eval_count <= self.warmup:
            return False
        
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            # For loss: lower is better
            improved = current_value < (self.best_value - self.min_delta)
        else:
            # For AUC: higher is better
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
        self.eval_count = 0

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Set the global seed
GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)
print(f"Random seed set to: {GLOBAL_SEED}")


# Training
dataset = Dataset(
    feature_frame,
    positive_event_labels=[
        'Death', 'Revisit - AE', 'Revisit - GP', 'Revisit - Readmission', 'Treatment delay/dose reduction'
    ],
    minimum_input_window=datetime.timedelta(days=14),
    minimum_wearing_days=7,
    output_window=datetime.timedelta(weeks=4),
    max_sequence_length=1000,
)
dataset.set_labeling_strategy(strategy_name="binary")
# dataset.set_labeling_strategy(strategy_name="tls")

print("Number of patients in dataset:", len(dataset.useable_datetimes.keys()))

enable_plotting = True

# Training parameters
device = torch.device('cuda:0')
fraction_data = 1
num_bootstrap_iterations = 30
early_stopping_patience = 5
early_stopping_warmup = 2  # Number of initial evaluations to ignore before monitoring (useful to skip unstable early training)
evaluation_warmup = 5  # Number of initial epochs to skip before first evaluation
evaluation_interval = 40
test_sample_size = 150
use_early_stopping = False
gradient_clip_value = 0.3 
use_evaluation = False

clear_memory()

loss_records = []
all_models = []
all_datasets = []
all_test_predictions = []
all_test_labels = []
all_evaluation_epochs = []  # Track evaluation epochs for each bootstrap


# Training loop
for bootstrap_index in tqdm(range(num_bootstrap_iterations), desc="Bootstrap Iterations"):
    # Set different seed for each bootstrap iteration to get different samples
    # This ensures reproducibility while allowing different bootstrap samples
    bootstrap_seed = GLOBAL_SEED + bootstrap_index * 100  # Larger offset for better separation
    set_seed(bootstrap_seed)
    
    dataset = copy.deepcopy(dataset)  # Create a new instance for each bootstrap iteration
    dataset.scaling_simulation(scaling_fraction=fraction_data) if fraction_data < 1 else None
    dataset.train_test_split(percentage_training=0.8)
    training_data = dataset.train(num_epochs=epochs, batch_size=batch_size)
    num_labels = dataset.num_labels
    num_std_features = getattr(dataset, 'num_std_features', 0)  # Get number of standard features
    all_datasets.append(copy.deepcopy(dataset))

    # model = Model(hidden_dim=hidden_dim, num_types=num_labels, num_std_features=num_std_features).to(device)
    model = Model(
        num_types=num_labels, 
        hidden_dim=hidden_dim, 
        num_std_features=num_std_features,
        # num_layers=num_layers,
        # num_heads=num_heads,
        # dropout=dropout,
        # attention_dropout=attention_dropout
    ).to(device)
    
    # Apply model-specific seed for weight initialization (different from data seed)
    model_seed = bootstrap_seed + 10000  # Large offset to avoid overlap with data seeds
    set_seed(model_seed)
    
    # Re-initialize model weights with the new seed
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.RNN):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
    
    model.apply(init_weights)
    all_models.append(model)
    if bootstrap_index == 0:
        print(f"Model created with {num_std_features} standard features")
        print(summary(model, input_data=[
            torch.randn(batch_size, 10, device=device),  # value
            torch.randint(0, num_labels, (batch_size, 10), device=device),  # type
            torch.ones(batch_size, 10, dtype=torch.int32, device=device),  # observation_mask
            torch.randn(batch_size, num_std_features, device=device) if num_std_features > 0 else None  # std_input
        ]))

    # Use AdamW optimizer as per STraTS original implementation
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Set seed for training loop to ensure reproducible batch ordering within each bootstrap
    training_seed = bootstrap_seed + 20000  # Another large offset for training
    set_seed(training_seed)
    
    # Initialize early stopping (monitor test loss with warmup to skip initial unstable evaluations)
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=0.01, 
        mode='max', 
        warmup=early_stopping_warmup
    )
    
    model.train()

    loss_history = {'training': [], 'testing': [], 'auc': []}
    evaluation_epochs = []
    final_probs = None
    final_labels = None

    for epoch_index, epoch_data in enumerate(training_data):
        inputs = {key: value.to(device) for key, value in epoch_data['input'].items()}
        std_inputs = epoch_data['std_input'].to(device)
        targets = epoch_data['output'].to(device)

        optimizer.zero_grad()
        predictions = model(**inputs, std_input=std_inputs)

        train_loss = loss_function(predictions, targets.float())
        train_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
        
        optimizer.step()

        loss_history['training'].append(train_loss.item())
        inputs = None  # Release input tensors
        clear_memory()

        # Evaluate model every 'evaluation_interval' epochs
        if ((epoch_index + 1) > evaluation_warmup and (epoch_index + 1) % evaluation_interval == 0) or (epoch_index == epochs - 1):
            evaluation_epochs.append(epoch_index + 1)
            test_loss, test_auc, probs, labels = evaluate_model(model, dataset, device, test_sample_size=test_sample_size)
            loss_history['testing'].append(test_loss)
            loss_history['auc'].append(test_auc)
            print(f"epoch {epoch_index+1}: test AUC:{test_auc:.4f}, test loss: {test_loss:.4f}")
            final_probs = probs
            final_labels = labels
            
            # Check for early stopping (monitor test loss for overfitting)
            if use_early_stopping and early_stopping(test_auc, epoch_index + 1):
                print(f"Early stopping triggered at epoch {epoch_index+1}")
                break

    all_test_predictions.append(final_probs)
    all_test_labels.append(final_labels)
    loss_records.append(loss_history)
    all_evaluation_epochs.append(evaluation_epochs)  # Save evaluation epochs for this bootstrap
    clear_memory()

# Compute mean and confidence intervals
# Handle variable-length sequences due to early stopping
# Find the maximum length for training losses
max_train_length = max(len(record['training']) for record in loss_records)
max_test_length = max(len(record['testing']) for record in loss_records)

# Pad shorter sequences with NaN
train_losses_padded = []
for record in loss_records:
    padded = np.full(max_train_length, np.nan)
    padded[:len(record['training'])] = record['training']
    train_losses_padded.append(padded)

test_losses_padded = []
for record in loss_records:
    padded = np.full(max_test_length, np.nan)
    padded[:len(record['testing'])] = record['testing']
    test_losses_padded.append(padded)

auc_scores_padded = []
for record in loss_records:
    padded = np.full(max_test_length, np.nan)
    padded[:len(record['auc'])] = record['auc']
    auc_scores_padded.append(padded)

# Convert to numpy arrays
train_losses = np.array(train_losses_padded)
test_losses = np.array(test_losses_padded)
auc_scores = np.array(auc_scores_padded)

# Compute statistics using nanmean/nanstd to ignore NaN values
train_mean, test_mean = np.nanmean(train_losses, axis=0), np.nanmean(test_losses, axis=0)
train_std, test_std = np.nanstd(train_losses, axis=0), np.nanstd(test_losses, axis=0)

# Count valid (non-NaN) values for proper confidence interval calculation
train_counts = np.sum(~np.isnan(train_losses), axis=0)
test_counts = np.sum(~np.isnan(test_losses), axis=0)

train_ci = 1.96 * train_std / np.sqrt(train_counts)
test_ci = 1.96 * test_std / np.sqrt(test_counts)

auc_mean, auc_std = np.nanmean(auc_scores, axis=0), np.nanstd(auc_scores, axis=0)
auc_counts = np.sum(~np.isnan(auc_scores), axis=0)
auc_ci = 1.96 * auc_std / np.sqrt(auc_counts)

# Function to plot curves with confidence intervals and optional individual curves
def plot_with_ci(x, mean, ci, individual_curves=None, color='blue', label=''):
    if individual_curves is not None:
        for curve in individual_curves:
            # Plot each curve with its actual length
            x_curve = x[:len(curve)] if len(curve) <= len(x) else x
            plt.plot(x_curve, curve[:len(x_curve)], color=color, alpha=0.15)

    plt.plot(x, mean, label=label, color=color, linewidth=2)
    plt.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.3)

# Prepare x-axis values
epochs_range = np.arange(1, len(train_mean) + 1)

# Extract individual curve lists
individual_train_losses = [record['training'] for record in loss_records]
individual_test_losses = [record['testing'] for record in loss_records]
individual_aucs = [record['auc'] for record in loss_records]

# Create a common evaluation epoch grid (take the longest one)
max_evaluation_epochs = max(all_evaluation_epochs, key=len)

# === Plot AUC Curves ===
plt.figure(figsize=(5, 4))
plot_with_ci(max_evaluation_epochs, auc_mean, auc_ci, individual_curves=individual_aucs, color='green', label='AUC Mean ± 95% CI')
plt.axhline(y=np.nanmax(auc_mean), color='orange', linestyle='--', label=f'Best AUC: {np.nanmax(auc_mean):.2f}')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing')
plt.axhline(y=0.75, color='g', linestyle='--', label='75')
plt.xlabel("Epochs", fontsize=14, fontweight='bold')
plt.ylabel("AUC", fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.title(f"AUC Visualization ({num_bootstrap_iterations} Bootstrap Iterations)", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10)
# prepared result folder
result_folder = f"../../Results/results_{np.nanmean([record['auc'][-1] for record in loss_records]):.2f}_{num_bootstrap_iterations}bs_{epochs}ep_{hidden_dim}hd_{batch_size}_seed{GLOBAL_SEED}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
plt.savefig(os.path.join(result_folder, "auc_curves.svg"), format='svg')
plt.show()


# === Plot Loss Curves ===
plt.figure(figsize=(5, 4))
plot_with_ci(epochs_range, train_mean, train_ci, individual_curves=individual_train_losses, color='blue', label='Training Loss Mean ± 95% CI')
plot_with_ci(max_evaluation_epochs, test_mean, test_ci, individual_curves=individual_test_losses, color='red', label='Testing Loss Mean ± 95% CI')
plt.xlabel("Epochs", fontsize=14, fontweight='bold')
plt.ylabel("Loss", fontsize=14, fontweight='bold')
plt.title(f"Loss Visualization ({num_bootstrap_iterations} Bootstrap Iterations)", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10)
plt.savefig(os.path.join(result_folder, "loss_curves.svg"), format='svg')
plt.show()


# === Plot ROC Curve at Final Epoch ===
tpr_list = []
fpr_list = []
individual_auc_list = []

for probs, labels in zip(all_test_predictions, all_test_labels):
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        # Calculate AUC for this bootstrap
        individual_auc = roc_auc_score(labels, probs)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        individual_auc_list.append(individual_auc)

# Interpolate TPRs to a common FPR grid for averaging
common_fpr = np.linspace(0, 1, 100)
interp_tprs = []

for fpr, tpr in zip(fpr_list, tpr_list):
    interp_tpr = np.interp(common_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

mean_tpr = np.mean(interp_tprs, axis=0)
std_tpr = np.std(interp_tprs, axis=0)
ci_tpr = 1.96 * std_tpr / np.sqrt(len(interp_tprs))

# Calculate mean AUC from individual bootstrap AUCs (more accurate than from training)
mean_auc_from_roc = np.mean(individual_auc_list)
std_auc_from_roc = np.std(individual_auc_list)
ci_auc_from_roc = 1.96 * std_auc_from_roc / np.sqrt(len(individual_auc_list))

# Plot the ROC Curve
plt.figure(figsize=(5, 5))
plt.plot(common_fpr, mean_tpr, color='blue', label=f'Mean ROC\nAUC={mean_auc_from_roc:.2f}±{ci_auc_from_roc:.2f}')
plt.fill_between(common_fpr, mean_tpr - ci_tpr, mean_tpr + ci_tpr, color='blue', alpha=0.3, label='95% CI')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.ylim(0, 1)
plt.xlim(0, 1)
# plt.title('ROC Curve at Final Epoch with 95% CI', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='lower right')
plt.grid(False)
plt.savefig(os.path.join(result_folder, "roc_curve.svg"), format='svg')
plt.show()

# === Plot Precision-Recall Curve (AURPC) at Final Epoch ===
precision_list = []
recall_list = []
ap_list = []

for probs, labels in zip(all_test_predictions, all_test_labels):
    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(ap)

# Interpolate precision to a common recall grid for averaging
common_recall = np.linspace(0, 1, 100)
interp_precisions = []

for precision, recall in zip(precision_list, recall_list):
    interp_precision = np.interp(common_recall, recall[::-1], precision[::-1])
    interp_precisions.append(interp_precision)

mean_precision = np.mean(interp_precisions, axis=0)
std_precision = np.std(interp_precisions, axis=0)
ci_precision = 1.96 * std_precision / np.sqrt(len(interp_precisions))

mean_ap = np.mean(ap_list) if ap_list else float('nan')
std_ap = np.std(ap_list) if ap_list else float('nan')
ci_ap = 1.96 * std_ap / np.sqrt(len(ap_list)) if ap_list else float('nan')

# Plot the Precision-Recall Curve
plt.figure(figsize=(5, 5))
plt.plot(common_recall, mean_precision, color='purple', label=f'Mean PR\nAP={mean_ap:.2f}±{ci_ap:.2f}')
plt.fill_between(common_recall, mean_precision - ci_precision, mean_precision + ci_precision, color='purple', alpha=0.3, label='95% CI')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
# plt.title('Precision-Recall Curve at Final Epoch with 95% CI', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig(os.path.join(result_folder, "precision_recall_curve.svg"), format='svg')
plt.show()

# pickle.dump([all_models, all_datasets], open(os.path.join(result_folder, "models_and_datasets.pkl"), "wb"))
clear_memory()


# Visualising Feature Importance from Attention Weights
feature_weight_total = {}
feature_count = {}

for model, dataset in zip(all_models, all_datasets):
    model.eval()
    with torch.no_grad():
        interpretation_data = dataset.test(batch_size=150)
        inputs = {k: v.to(device) for k, v in interpretation_data['input'].items()}
        std_inputs = interpretation_data['std_input'].to(device)
        output, attention_weights = model(
            inputs['value'],
            inputs['type'],
            inputs['observation_mask'],
            std_input=std_inputs,
            return_attention=True,
        )

        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        labels = interpretation_data['output'].squeeze().cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()
        mask = inputs['observation_mask'].cpu().numpy()

        # Create back conversion mapping from index to label
        # Note: Index 0 is padding, we need to handle this special case
        back_conversion = {}
        for k, v in dataset.label_to_idx.items():
            back_conversion[v] = k
        # Handle padding index
        back_conversion[0] = "PADDING"  # Or we can skip padding

        for patient_index, patient_weight in enumerate(attention_weights):
            for i, weight in enumerate(patient_weight):
                # Check if this position is masked (i.e., whether it's valid data)
                if mask[patient_index][i] == 0:
                    continue  # Skip padding/masked positions
                    
                feature_value = interpretation_data['input']['type'][patient_index][i]
                feature_idx = feature_value.item()
                
                # Skip padding index
                if feature_idx == 0:
                    continue
                
                # Get feature type name
                if feature_idx in back_conversion:
                    feature_type = back_conversion[feature_idx]
                    feature_weight_total[feature_type] = feature_weight_total.get(feature_type, 0) + weight.item()
                    feature_count[feature_type] = feature_count.get(feature_type, 0) + 1
                else:
                    print(f"Warning: Unknown feature index {feature_idx}")

feature_weight_avg = {k: feature_weight_total[k] / feature_count[k] for k in feature_weight_total.keys()}

# Separate features by modality
event_modality_features = [
    'Chemotherapy', 'Revisit - AE', 'Revisit - GP', 'Revisit - Readmission', 'Adverse Event Severity', 'Treatment delay/dose reduction',
    'Mixed therapy', 'Immunotherapy', 'Death', 'Hormone therapy', 'Other therapy'
]

event_features = {} 
timeseries_features = {}

# Categorize features
for feature, value in feature_weight_avg.items():
    if feature in event_modality_features:
        event_features[feature] = value
    else:
        timeseries_features[feature] = value

# Sort features within each modality
sorted_event_features = sorted(event_features.items(), key=lambda x: x[1], reverse=True)
sorted_ts_features = sorted(timeseries_features.items(), key=lambda x: x[1], reverse=True)

# Function to plot feature importance for a specific modality
def plot_feature_importance(sorted_features, title, color, top_n=6, display_others=True):
    # Number of top features to display individually
    top_n = min(top_n, len(sorted_features))
    # top_n = len(sorted_features) # show all features
    
    # Separate top N features and combine the rest
    top_features = sorted_features[:top_n]
    other_features = sorted_features[top_n:]
    other_value = sum(f[1] for f in other_features) if other_features else 0
    
    # Create display data with top features and "Others" category
    display_names = [f[0] for f in top_features]
    display_values = [f[1] for f in top_features]
    
    # Add "Others" category if there are more than top_n features
    if other_features and display_others:
        display_names.append(f"{len(other_features)} other items (sum)")
        display_values.append(other_value)
    
    # Plotting
    plt.figure(figsize=(5, 4))
    bars = plt.barh(display_names, display_values, color=color)
    if other_features and display_others:
        bars[-1].set_color('lightgray')  # Color the "Others" bar differently
    
    plt.xlabel('Average Attention Weight per Instance', fontsize=12)
    # plt.title(f'{title} Feature Importance from Attention Weights')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.xticks(fontsize=11)
    plt.xlim(0, 0.004)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"{title.lower().replace(' ', '_')}_feature_importance.svg"), format='svg')
    plt.show()

    
    # Print the breakdown of "Others" category for reference
    if other_features:
        print(f"\nBreakdown of '{len(other_features)} other items' for {title}:")
        for name, value in other_features:
            print(f"{name}: {value:.5f}")

# Plot event modality features
plot_feature_importance(sorted_event_features, "Event Modality", "lightsalmon", top_n=6, display_others=False)

# Plot time series modality features
plot_feature_importance(sorted_ts_features, "Time Series Modality", "skyblue", top_n=6, display_others=False)

clear_memory()

# Function to create improved visualization
def create_enhanced_risk_plot(patient_id, all_predictions, all_ground_truths, feature_frame):
    """
    Create an enhanced visualization of risk predictions with clinical events.
    
    Args:
        patient_id: ID of the patient
        all_predictions: Dictionary mapping timestamps to lists of predicted risks
        all_ground_truths: Dictionary mapping timestamps to ground truth values
        feature_frame: FeatureFrame containing patient data
    """
    if not all_predictions:
        print(f"No prediction data available for patient {patient_id}")
        return
        
    # Create figure with two rows in a single plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Get all clinical events
    events_df = feature_frame.ts_features[
        (feature_frame.ts_features['label'].isin(['Revisit - Readmission', 'Revisit - AE', 
                                              'Treatment delay/dose reduction', 'Death'])) &
        (feature_frame.ts_features['Patient ID'] == patient_id)
    ]
    
    # Process heart rate and wellness data for context
    hr_data = feature_frame.ts_features[
        (feature_frame.ts_features['label'] == 'HR-avg') & 
        (feature_frame.ts_features['Patient ID'] == patient_id)
    ]
    wellness_data = feature_frame.ts_features[
        (feature_frame.ts_features['label'] == 'Wellness check-in') & 
        (feature_frame.ts_features['Patient ID'] == patient_id)
    ]
    
    # Sort timestamps for proper plotting
    sorted_timestamps = sorted(all_predictions.keys())
    
    # Get predictions and ground truths in order
    avg_preds = [np.mean(all_predictions[ts]) for ts in sorted_timestamps]
    std_preds = [np.std(all_predictions[ts]) for ts in sorted_timestamps]
    gts = [all_ground_truths[ts][0] for ts in sorted_timestamps]
    
    # Calculate confidence intervals using t-distribution
    confidence_intervals = []
    for ts in sorted_timestamps:
        values = all_predictions[ts]
        n = len(values)
        if n > 1:  # Need at least 2 samples for confidence interval
            mean = np.mean(values)
            std_err = sem(values)
            conf_interval = t.interval(0.95, n-1, loc=mean, scale=std_err)
            confidence_intervals.append(conf_interval)
        else:
            # If only one prediction, use a dummy confidence interval
            mean = values[0]
            confidence_intervals.append((mean, mean))
    
    # === First subplot: Risk predictions with confidence interval ===
    # Add threshold line first (so it appears behind the shading)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Risk Threshold')
    
    # Create a step function version of ground truth that matches the plotting style
    # This ensures the shading aligns perfectly with the step function visualization
    x_extended = []
    y_extended = []
    
    # Add first point
    x_extended.append(sorted_timestamps[0])
    y_extended.append(gts[0])
    
    # Add step points (each x value appears twice - once for the end of previous step,
    # once for beginning of next step)
    for i in range(1, len(sorted_timestamps)):
        x_extended.append(sorted_timestamps[i])  # End of previous step
        x_extended.append(sorted_timestamps[i])  # Start of next step
        y_extended.append(gts[i-1])  # Previous value (for end of step)
        y_extended.append(gts[i])    # Current value (for start of step)
    
    # Add shading from ground truth to 0.5 threshold
    threshold_line = [0.5] * len(x_extended)
    ax1.fill_between(
        x_extended, 
        y_extended, 
        threshold_line,
        color='red', 
        alpha=0.15, 
        interpolate=True,
        label='Ground Truth'
    )
    
    # Plot the ground truth as a step function with clearer visualization
    # ax1.step(sorted_timestamps, gts, 'r-', label='Ground Truth', linewidth=2, where='post', alpha=0.7)
    
    # Plot the predicted risk with confidence interval
    ax1.plot(sorted_timestamps, avg_preds, 'b-', label='Predicted Risk', linewidth=2)
    ax1.fill_between(
        sorted_timestamps, 
        [ci[0] for ci in confidence_intervals], 
        [ci[1] for ci in confidence_intervals], 
        color='blue', alpha=0.2, label='95% CI'
    )
    
    # Plot data points with size reflecting prediction confidence
    for i, ts in enumerate(sorted_timestamps):
        confidence_range = confidence_intervals[i][1] - confidence_intervals[i][0]
        confidence_size = max(20, 50 * (1 - min(confidence_range, 0.5)/0.5))  # Larger points for higher confidence
        ax1.scatter(ts, avg_preds[i], s=confidence_size, color='blue', alpha=0.7)
    
    # Add labels and title
    ax1.set_ylabel('Risk Score', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_title(f'{patient_id}', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === Second subplot: Clinical events timeline ===
    # Create clinical events timeline
    event_types = {
        'Revisit - Readmission': {'color': 'red', 'marker': 'v', 'markersize': 12},
        'Revisit - AE': {'color': 'orange', 'marker': 'v', 'markersize': 12},
        'Treatment delay/dose reduction': {'color': 'purple', 'marker': 'x', 'markersize': 12},
        'Death': {'color': 'black', 'marker': 'X', 'markersize': 15},
    }
    
    # Set up y-positions for different event types
    y_positions = {event: idx for idx, event in enumerate(event_types.keys(), start=1)}
    max_y = len(event_types) + 1
    
    # Plot each event type
    for event_type, properties in event_types.items():
        event_data = events_df[events_df['label'] == event_type]
        if not event_data.empty:
            timestamps = event_data['start'].values
            ax2.plot(
                timestamps, 
                [y_positions[event_type]] * len(timestamps),
                linestyle='', 
                marker=properties['marker'], 
                color=properties['color'],
                markersize=properties['markersize'], 
                label=event_type
            )
    
    # Configure the event timeline axis
    ax2.set_yticks(list(y_positions.values()))
    ax2.set_yticklabels(list(y_positions.keys()), fontsize=10)
    ax2.set_ylim(0.5, max_y + 0.5)
    ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Events', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=11)
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Format x-axis for both plots
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    
    return fig, ax1, ax2

train_patients, test_patients = all_datasets[-1].split
all_patients = train_patients + test_patients


for i in range(len(all_patients)):
    target_patient_id = all_patients[i]
    print(f"\nProcessing patient {target_patient_id} ({i+1}/{len(all_patients)})")
    
    # Collect predictions from all bootstrap runs where patient is in test set
    all_predictions = {}  # {timestamp: [predictions from bootstraps]}
    all_ground_truths = {}  # {timestamp: [ground truth values]}

    for model_idx, (model, dataset) in enumerate(zip(all_models, all_datasets)):
        train_patients, test_patients = dataset.split
        
        # Skip if patient is in training set for this bootstrap
        if target_patient_id not in test_patients:
            continue
        
        try:
            # Get inference data for this patient
            result = dataset.inference_single(target_patient_id)
            
            # Extract predictions
            model.eval()
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in result['input'].items()}
                std_inputs = result['std_input'].to(device)
                output = model(**inputs, std_input=std_inputs)
                pred_probs = torch.sigmoid(output).cpu().numpy().flatten()
            
            # Store prediction data by timestamp
            for idx, ts in enumerate(result['timestamp']):
                if ts not in all_predictions:
                    all_predictions[ts] = []
                    all_ground_truths[ts] = []
                all_predictions[ts].append(pred_probs[idx])
                all_ground_truths[ts].append(result['output'].cpu().numpy().flatten()[idx])
            
        except Exception as e:
            print(f"Error in bootstrap {model_idx}: {str(e)}")
    
    if not all_predictions:
        print(f"No predictions available for patient {target_patient_id}. Skipping.")
        continue
    
    print(f"Found {len(all_predictions)} prediction timestamps")
    
    # Use the original feature_frame from training/prediction
    single_patient_ff = feature_frame
    
    # Verify that the patient exists in the original feature_frame
    if target_patient_id not in single_patient_ff.ts_features['Patient ID'].values:
        print(f"Warning: Patient {target_patient_id} not found in original feature_frame. Skipping.")
        continue
    
    # Create and show the enhanced visualization
    fig, ax1, ax2 = create_enhanced_risk_plot(
        target_patient_id, 
        all_predictions, 
        all_ground_truths, 
        single_patient_ff
    )
    
    os.makedirs(os.path.join(result_folder, "patient_visualizations"), exist_ok=True)
    fig.savefig(os.path.join(result_folder, "patient_visualizations", f"{target_patient_id}_risk_plot.svg"), format='svg')
    plt.show()

