import torch
from transformers import RobertaModel
from transformers import RobertaConfig
from transformers import RoFormerModel
from transformers import RoFormerConfig
from transformers.utils import ModelOutput
from torch.nn import MSELoss
from torch import nn
from transformers import AdamW
from tqdm import tqdm
import numpy as np
# from accelerate import Accelerator
from transformers.optimization import get_constant_schedule_with_warmup
from transformers.trainer_utils import set_seed
import time
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import random
import os
from typing import Optional, Tuple, Union
from matplotlib import pyplot as plt
import pickle
from matplotlib.lines import Line2D
import scipy
from itertools import accumulate

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_value)
    
def get_last_epoch(ckpt_path):
    all_files = os.listdir(ckpt_path)
    all_files = sorted([int(k[6:].split('.')[0]) for k in all_files])
    last_epoch = all_files[-1]
    return last_epoch
    
    
def linear_alpha_scheduler(min_alpha, max_alpha, total_epochs, current_epoch):
    """
    Computes alpha scaling linearly from min_alpha to max_alpha over total_epochs and returns alpha and r² (1 / (1 + alpha²))
    """
    # Calculate the change in alpha per epoch
    alpha_increment_per_epoch = (max_alpha - min_alpha) / total_epochs

    # Calculate current alpha based on the linear progression
    current_alpha = min_alpha + (alpha_increment_per_epoch * current_epoch)

    # Calculate r^2 value based on current alpha
    current_r2 = 1 / (1 + current_alpha**2)

    return current_alpha, current_r2


def plot_scatter_between_matrices(tensor1, tensor2):
    """
    Plots scatterplots of two tensors [N,D] in separate subplots for each dimension
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("The shapes of the two tensors must be the same")

    num_points, num_dims = tensor1.shape
    cmap = plt.cm.get_cmap('viridis', num_dims)
    fig, axes = plt.subplots(1, num_dims, figsize=(15, 5))   
    for i in range(num_dims):
        axes[i].scatter(tensor1[:, i], tensor2[:, i], color=cmap(i), alpha=0.5)
        axes[i].set_xlabel(f'Prediction Dim {i+1}')
        axes[i].set_ylabel(f'Ground_truth Dim {i+1}')
        axes[i].set_title(f'Dimension {i+1} Scatter Plot')
        pr2 = scipy.stats.pearsonr(tensor2[:, i], tensor1[:, i])[0]
        r2_value = pr2*pr2
        axes[i].text(0.05, 0.95, f'R^2: {r2_value:.2f}', transform=axes[i].transAxes, fontsize=9, verticalalignment='top')

    return fig, axes

def plot_histogram_pr2s_between_matrices(tensor1, tensor2):
    """
    Plots histogram of pr2s for two tensors[N,D]
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("The shapes of the two tensors must be the same")
        
    num_points, num_dims = tensor1.shape
    pr2s = []
    for i in range(num_dims):
        pr2 = scipy.stats.pearsonr(tensor2[:, i], tensor1[:, i])[0]
        r2_value = pr2*pr2
        pr2s.append(r2_value)
    return torch.tensor(pr2s)

def pearson_r_squared_between_matrices(mat1, mat2):
    assert mat1.shape == mat2.shape
    
    mean1 = torch.mean(mat1, 1)
    mean2 = torch.mean(mat2, 1)

    # Mean-centering the columns
    mat1_centered = mat1 - mean1.unsqueeze(1)
    mat2_centered = mat2 - mean2.unsqueeze(1)

    # Compute Pearson correlation coefficient for each column
    numerator = (mat1_centered * mat2_centered).sum(dim=0)
    denominator = torch.sqrt((mat1_centered ** 2).sum(dim=0) * (mat2_centered ** 2).sum(dim=0))

    # Avoid division by zero
    valid = denominator != 0
    r = torch.zeros_like(denominator)
    r[valid] = numerator[valid] / denominator[valid]

    # Squaring the Pearson correlation coefficient
    r_squared = r ** 2

    return r_squared

def rowwise_pearson_r2(a, b):
    if torch.is_tensor(a): a = a.cpu().numpy()
    if torch.is_tensor(b): b = b.cpu().numpy()

    r2 = np.array([pearsonr(x, y)[0]**2 for x, y in zip(a, b)])
    return r2.mean()

def get_noise_percentage(noise_percentage_type, epoch, epochs, noise_percentage):
    if noise_percentage_type == 'constant':
        return noise_percentage
    
def get_pca_labels(embs, scaler, ipca):
    embs = scaler.transform(embs)
    embs = ipca.transform(embs)
    return torch.tensor(embs, dtype = torch.float32)

def get_emb_scaler(embs, scaler):
    embs = scaler.transform(embs)
    return torch.tensor(embs, dtype = torch.float32)

class PCAGenomeDataset(Dataset):
    def __init__(self, num_samples,id_vs_num_filepath = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/dataset_final/train_proteins.csv',\
                 parent_directory = "/orange/juannanzhou/bacteria_genome/protein_faa_files/sorted_esm_embeds/480_dim", max_seq_len = 5000, pad = True,\
                 return_original_len = False):
        self.max_seq_len = max_seq_len
        self.id_vs_num = pd.read_csv(id_vs_num_filepath)
        self.parent_directory = str(parent_directory)
        self.num_samples = num_samples
        self.pad = pad
        self.scaler_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/scaler_onlyaa.pkl'
        self.pca_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/pca_3dim_onlyaa.pkl'
        self.pca_scaler_for_labels_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/pca_scaler_for_labels_onlyaa.pkl'
        self.return_original_len = return_original_len
        with open(self.scaler_path,'rb') as f:
            self.scaler = pickle.load(f)
        with open(self.pca_path,'rb') as f:
            self.ipca = pickle.load(f)
        with open(self.pca_scaler_for_labels_path,'rb') as f:
            self.pca_scaler_for_labels_path = pickle.load(f)

    def __len__(self):
        if self.num_samples > 0:
            return self.num_samples
        if self.num_samples == -1:
            return len(self.id_vs_num)

    def __getitem__(self, idx):
        self.pt_file_path = os.path.join(self.parent_directory, 'onlyaa_mean_sorted_'+self.id_vs_num['Genome_ID'].iloc[idx].split('.')[0]+'.pt')
        self.emb = torch.load(self.pt_file_path, map_location = 'cpu')
        self.original_length = self.emb.shape[0]
        if self.emb.shape[0] > 5000:
            self.emb = self.emb[:5000,:]
        # self.labels = get_pca_labels(self.emb, self.scaler, self.ipca)
        # self.labels = get_emb_scaler(self.labels, self.pca_scaler_for_labels_path)
        # self.emb = get_emb_scaler(self.emb, self.scaler)
        # for all pcs only
        self.emb = get_emb_scaler(self.emb, self.scaler)
        self.labels = self.emb
        self.attention_mask = torch.randint(1,2,(self.emb.shape[0],))
        if self.emb.shape[0] < self.max_seq_len and self.pad==True:
            self.attention_mask = F.pad(self.attention_mask, (0, self.max_seq_len-self.emb.shape[0]), 'constant', 0)
            self.emb = F.pad(self.emb, (0, 0, 0, self.max_seq_len-self.emb.shape[0]), 'constant', 0.0)
            self.labels = F.pad(self.labels, (0, 0, 0, self.max_seq_len-self.labels.shape[0]), 'constant', 0.0)
        

        if self.return_original_len:
            return self.emb, self.labels, self.attention_mask, self.original_length
        else:
            return self.emb, self.labels, self.attention_mask
        
class ContigGenomeDatasetOld(Dataset):
    def __init__(self, num_samples,id_vs_num_filepath = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/dataset_final/train_proteins_with_contig_len.csv',\
                 parent_directory = "/orange/juannanzhou/bacteria_genome/protein_faa_files/sorted_esm_embeds/480_dim", max_seq_len = 50, pad = True):
        self.max_seq_len = max_seq_len
        self.id_vs_num = pd.read_csv(id_vs_num_filepath)
        self.parent_directory = str(parent_directory)
        self.num_samples = num_samples
        self.pad = pad
        self.scaler_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/scaler_onlyaa.pkl'
        self.pca_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/pca_3dim_onlyaa.pkl'
        self.pca_scaler_for_labels_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/pca_scaler_for_labels_onlyaa.pkl'
        with open(self.scaler_path,'rb') as f:
            self.scaler = pickle.load(f)
        with open(self.pca_path,'rb') as f:
            self.ipca = pickle.load(f)
        with open(self.pca_scaler_for_labels_path,'rb') as f:
            self.pca_scaler_for_labels_path = pickle.load(f)
        self.prefix_sums = list(accumulate(self.id_vs_num.num_50l_contigs))
        self.contig_len=50

    def __len__(self):
        if self.num_samples > 0:
            return self.num_samples
        if self.num_samples == -1:
            return sum(self.id_vs_num.num_50l_contigs)
        
    def get_genomome_idx(self, idx):
        for i, s in enumerate(self.prefix_sums):
            if s >= idx:
                return i
    
    def get_contig_index(self, idx):#RANDOM CONTIG SAMPLING
        contig_start = random.randint(0, self.id_vs_num.iloc[idx].num_50l_contigs)
        return contig_start, contig_start+self.contig_len

    def __getitem__(self, idx):
        idx = self.get_genomome_idx(idx)
        contig_start, contig_end = self.get_contig_index(idx)
        self.pt_file_path = os.path.join(self.parent_directory, 'onlyaa_mean_sorted_'+self.id_vs_num['Genome_ID'].iloc[idx].split('.')[0]+'.pt')
        self.emb = torch.load(self.pt_file_path, map_location = 'cpu')[contig_start:contig_end,:]
        # for all pcs only
        self.emb = get_emb_scaler(self.emb, self.scaler)
        self.labels = self.emb
        self.attention_mask = torch.randint(1,2,(self.emb.shape[0],))
        if self.emb.shape[0] < self.max_seq_len and self.pad==True:
            self.attention_mask = F.pad(self.attention_mask, (0, self.max_seq_len-self.emb.shape[0]), 'constant', 0)
            self.emb = F.pad(self.emb, (0, 0, 0, self.max_seq_len-self.emb.shape[0]), 'constant', 0.0)
            self.labels = F.pad(self.labels, (0, 0, 0, self.max_seq_len-self.labels.shape[0]), 'constant', 0.0)
        return self.emb, self.labels, self.attention_mask
    
class ContigGenomeDataset(Dataset):
    def __init__(self, num_samples,
                 id_vs_num_filepath='/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/dataset_final/train_proteins_with_contig_len.csv',
                 parent_directory="/orange/juannanzhou/bacteria_genome/protein_faa_files/sorted_esm_embeds/480_dim",
                 max_seq_len=50, pad=True):
        self.max_seq_len = max_seq_len
        self.id_vs_num = pd.read_csv(id_vs_num_filepath)
        self.parent_directory = str(parent_directory)
        self.num_samples = num_samples
        self.pad = pad
        self.scaler_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/scaler_onlyaa.pkl'
        self.pca_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/pca_3dim_onlyaa.pkl'
        self.pca_scaler_for_labels_path = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/norm_5klength/pca_scaler_for_labels_onlyaa.pkl'
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(self.pca_path, 'rb') as f:
            self.ipca = pickle.load(f)
        with open(self.pca_scaler_for_labels_path, 'rb') as f:
            self.pca_scaler_for_labels_path = pickle.load(f)
        self.prefix_sums = list(accumulate(self.id_vs_num.num_50l_contigs))
        self.prefix_sums = [0] + self.prefix_sums  # e.g., [0, 50, 110, ...]
        self.contig_len = 50

    def __len__(self):
        if self.num_samples > 0:
            return self.num_samples
        elif self.num_samples == -1:
            return self.prefix_sums[-1]

    def get_genomome_idx(self, idx):
        # Returns the genome index such that:
        # prefix_sums[i] <= idx < prefix_sums[i+1]
        for i in range(len(self.prefix_sums) - 1):
            if self.prefix_sums[i] <= idx < self.prefix_sums[i + 1]:
                return i
        return len(self.prefix_sums) - 2  # Fallback: last genome

    def get_contig_index(self, genome_index, local_idx):
        # Deterministic: local_idx is the contig index within the genome
        contig_start = local_idx
        return contig_start, contig_start + self.contig_len

    def __getitem__(self, idx):
        genome_index = self.get_genomome_idx(idx)
        local_idx = idx - self.prefix_sums[genome_index]
        contig_start, contig_end = self.get_contig_index(genome_index, local_idx)

        pt_file_path = os.path.join(self.parent_directory,
                                    'onlyaa_mean_sorted_' +
                                    self.id_vs_num['Genome_ID'].iloc[genome_index].split('.')[0] + '.pt')
        emb = torch.load(pt_file_path, map_location='cpu')
        emb = emb[contig_start:contig_end, :]
        emb = get_emb_scaler(emb, self.scaler)
        labels = emb
        attention_mask = torch.ones(emb.shape[0], dtype=torch.int)
        if emb.shape[0] < self.max_seq_len and self.pad:
            attention_mask = F.pad(attention_mask, (0, self.max_seq_len - emb.shape[0]), 'constant', 0)
            emb = F.pad(emb, (0, 0, 0, self.max_seq_len - emb.shape[0]), 'constant', 0.0)
            labels = F.pad(labels, (0, 0, 0, self.max_seq_len - labels.shape[0]), 'constant', 0.0)
        return emb, labels, attention_mask
        
class MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    
def infer_model(model, inputs_embeds, attention_mask):
    outputs = model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds#.unsqueeze(dim = 0)
            )

    predicted_embeds = outputs
    return MaskedLMOutput(
            last_hidden_state=predicted_embeds,
        )

def get_output_and_loss(model, inputs_embeds, attention_mask, masked_tokens, labels):
    outputs = model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds#.unsqueeze(dim = 0)
            )

    predicted_embeds = outputs
    batch_size, max_seq_len, num_pc = predicted_embeds.shape
    
    # calculate MSE loss on masked tokens only
    loss_fct = MSELoss()
    masked_lm_loss = loss_fct(predicted_embeds[masked_tokens.squeeze(-1)], labels[masked_tokens.squeeze(-1)])
    total_loss = masked_lm_loss
    
    prediction_var = torch.var(predicted_embeds[masked_tokens.squeeze(-1)])
    label_var = torch.var(labels[masked_tokens.squeeze(-1)])
    prediction_mean = torch.mean(predicted_embeds[masked_tokens.squeeze(-1)])
    label_mean = torch.mean(labels[masked_tokens.squeeze(-1)])
    pr2 = torch.mean(pearson_r_squared_between_matrices(predicted_embeds[masked_tokens.squeeze(-1)], labels[masked_tokens.squeeze(-1)]))
    
    
    return MaskedLMOutput(
            loss=total_loss,
            last_hidden_state=predicted_embeds,
            prediction_mean = prediction_mean,
            prediction_var = prediction_var,
            label_var = label_var,
            label_mean= label_mean,
            masked_lm_loss = masked_lm_loss,
            last_hidden_state_masked = predicted_embeds[masked_tokens.squeeze(-1)],
            ground_truth_masked = labels[masked_tokens.squeeze(-1)],
            pr2 = pr2
        )

class BLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        if config.pca and config.pca_dim > 0:
            self.dense1 = nn.Linear(config.hidden_size, config.pca_dim)
        else:
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        x = self.dense1(features)
        return x
    
class MLP(nn.Module):
    """MLP before Roberta"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, features, **kwargs):
        x = self.dense(features)
        return x
    
class BacteriaLM(RobertaModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)


        if not add_pooling_layer:
            self.pooler = None  # This removes the pooling layer
            
        self.embeddings.word_embeddings = None
        
        self.mlp = MLP(config)
        self.lm_head = BLMHead(config)
        
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def forward(self, inputs_embeds, attention_mask):
        inputs_embeds = self.mlp(inputs_embeds)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        predicted_embeds = self.lm_head(sequence_output)
        return predicted_embeds
    
class BacteriaLM_alllayers(RobertaModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        # Modify or remove the pooling layer based on the add_pooling_layer flag
        if not add_pooling_layer:
            self.pooler = None  # This removes the pooling layer
            
        self.embeddings.word_embeddings = None
        # self.embeddings.token_type_embeddings  = None
        
        if config.attn_implementation == "flash_attention_2":
            for attention_layer in self.encoder.layer:
                attention_layer.attention.self = FlashRobertaSelfAttention(config)
        
        self.mlp = MLP(config)
        self.lm_head = BLMHead(config)
        
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def forward(self, inputs_embeds, attention_mask):
        inputs_embeds = self.mlp(inputs_embeds)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        # sequence_output = outputs.last_hidden_state
        predicted_embeds = self.lm_head(sequence_output)
        return predicted_embeds, outputs
    
class BacteriaLM_rope(RoFormerModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        # Modify or remove the pooling layer based on the add_pooling_layer flag
        if not add_pooling_layer:
            self.pooler = None  # This removes the pooling layer
            
        self.embeddings.word_embeddings = None
        # self.embeddings.token_type_embeddings  = None
        
        if config.attn_implementation == "flash_attention_2":
            for attention_layer in self.encoder.layer:
                attention_layer.attention.self = FlashRobertaSelfAttention(config)
        
        self.mlp = MLP(config)
        self.lm_head = BLMHead(config)
        
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def forward(self, inputs_embeds, attention_mask):
        inputs_embeds = self.mlp(inputs_embeds)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        # sequence_output = outputs.last_hidden_state
        predicted_embeds = self.lm_head(sequence_output)
        return predicted_embeds
    
class BacteriaLM_rope_all_layers(RoFormerModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        # Modify or remove the pooling layer based on the add_pooling_layer flag
        if not add_pooling_layer:
            self.pooler = None  # This removes the pooling layer
            
        self.embeddings.word_embeddings = None
        # self.embeddings.token_type_embeddings  = None
        
        if config.attn_implementation == "flash_attention_2":
            for attention_layer in self.encoder.layer:
                attention_layer.attention.self = FlashRobertaSelfAttention(config)
        
        self.mlp = MLP(config)
        self.lm_head = BLMHead(config)
        
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def forward(self, inputs_embeds, attention_mask):
        inputs_embeds = self.mlp(inputs_embeds)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        # sequence_output = outputs.last_hidden_state
        predicted_embeds = self.lm_head(sequence_output)
        return predicted_embeds, outputs
    
def get_attention_from_all_layer_model(model, attention_mask, inputs_embeds):
    predicted_embeds, outputs = model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds#.unsqueeze(dim = 0)
            )
    return outputs
    
