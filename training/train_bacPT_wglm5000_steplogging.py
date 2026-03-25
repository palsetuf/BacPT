import torch
import argparse
import pathlib
from transformers import RoFormerModel, RoFormerConfig
from torch.nn import MSELoss
from torch import nn
from transformers import AdamW
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers.optimization import get_constant_schedule_with_warmup
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.trainer_utils import set_seed
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random
from matplotlib import pyplot as plt
from collections import OrderedDict
import os
import sys
import gc
import math
sys.path.append("/blue/juannanzhou/palash.sethi/Projects/bacteria_genome")
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from model_training.final_training_scripts import utils as BLM

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", DeprecationWarning)

def cosine_mask_prob(t, t0, tfinal, max_prob=0.4, min_prob=0.01):
    ratio = (t - t0) / (tfinal - t0)
    return 0.5 * (1 + math.cos(math.pi * ratio)) * (max_prob - min_prob) + min_prob

def extend_context_length(old_state_dict_path, new_model_class, new_model_config):
    checkpoint_path = os.path.join(old_state_dict_path, 'checkpoints')
    last_epoch = BLM.get_last_epoch(checkpoint_path)
    checkpoint_path_epoch = os.path.join(checkpoint_path, 'model_' + str(last_epoch) + '.cp')
    checkpoint = torch.load(checkpoint_path_epoch, map_location='cpu')
    old_state_dict = OrderedDict((key.replace('module.', '',1), value) for key, value in checkpoint['model_state_dict'].items())
    # Remove position embeddings from old state dict
    modified_state_dict = {k: v for k, v in old_state_dict.items() 
                          if 'embed_positions.weight' not in k}
    
    # Initialize new model
    new_model = new_model_class(new_model_config)
    z,x = new_model.load_state_dict(modified_state_dict, strict=False)
    
    # Get position embeddings for interpolation
    old_pos_embed = old_state_dict['encoder.embed_positions.weight']
    
    # Interpolate position embeddings
    new_pos_embed = F.interpolate(
        old_pos_embed.unsqueeze(0).transpose(1, 2),
        size=new_model_config.max_position_embeddings,
        mode='linear'
    ).transpose(1, 2).squeeze(0)
    
    # Apply interpolated embeddings
    new_model.encoder.embed_positions.weight.data.copy_(new_pos_embed)
    
    # Verify weight transfer
    transfer_successful = True
    for key in old_state_dict:
        if 'embed_positions.weight' in key:
            continue
        if not torch.allclose(old_state_dict[key], new_model.state_dict()[key]):
            print(f"Warning: Weights not transferred correctly for {key}")
            transfer_successful = False
    
    if transfer_successful:
        print('All weights transferred successfully')
    
    return new_model

def evaluate_bacPT(noise_mean, noise_var, epoch, writer, model, test_dataloader, mask_prob, batch_size, accelerator, fixed_mask=None):
    model.eval()
    total_loss = 0
    total_prediction_mean = 0
    total_label_mean = 0
    total_prediction_var = 0
    total_label_var = 0
    total_pr2 = 0
    scaler = torch.cuda.amp.GradScaler()
    
    # Calculate global step for validation (assuming this is called after an epoch)
    global_step_val = epoch * len(test_dataloader)
    
    with torch.no_grad():
        for batch_num, (inputs_embeds, labels, attention_mask) in enumerate(test_dataloader):
            inputs_embeds, labels, attention_mask = inputs_embeds.to(accelerator.device),\
            labels.to(accelerator.device), attention_mask.to(accelerator.device)
            if fixed_mask:
                seed_value = 29122023
                set_seed(seed_value)
            rand = torch.rand(attention_mask.shape).to(accelerator.device)
            masked_tokens = (rand < mask_prob) & (attention_mask != 0)
            masked_tokens = torch.unsqueeze(masked_tokens, -1)
            mean = torch.zeros(noise_mean.shape).to(accelerator.device)
            std_dev = torch.tensor(noise_var** 0.5).to(accelerator.device)
            mean = mean.unsqueeze(0).unsqueeze(0).expand(inputs_embeds.size(0), inputs_embeds.size(1), -1)
            std_dev = std_dev.unsqueeze(0).unsqueeze(0).expand_as(mean)
            noise = torch.normal(mean = mean, std = std_dev).to(dtype=torch.float32)
            inputs_embeds  = torch.where(masked_tokens, noise, inputs_embeds)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = BLM.get_output_and_loss(model, inputs_embeds, attention_mask, masked_tokens, labels)
                loss = outputs.loss
                
                # Update totals for epoch-level metrics
                total_loss += loss.item()
                total_prediction_mean += outputs.prediction_mean.item()
                total_label_mean += outputs.label_mean.item()
                total_prediction_var += outputs.prediction_var.item()
                total_label_var += outputs.label_var.item()
                total_pr2 += outputs.pr2.item()
                
                # Log per-step metrics to tensorboard
                writer.add_scalar("Loss/test_step", loss.item(), global_step_val + batch_num)
                writer.add_scalar("Var/test_step/pred", outputs.prediction_var.item(), global_step_val + batch_num)
                writer.add_scalar("Var/test_step/label", outputs.label_var.item(), global_step_val + batch_num)
                writer.add_scalar("Mean/test_step/pred", outputs.prediction_mean.item(), global_step_val + batch_num)
                writer.add_scalar("Mean/test_step/label", outputs.label_mean.item(), global_step_val + batch_num)
                writer.add_scalar("pr2/test_step", outputs.pr2.item(), global_step_val + batch_num)
                
                # Store matrices for histogram
                m1, m2 = outputs.last_hidden_state_masked.cpu().numpy(), outputs.ground_truth_masked.cpu().numpy()
                
                # Log histograms selectively to avoid overwhelming tensorboard
                if batch_num % 10 == 0:
                    step_pr2s = BLM.plot_histogram_pr2s_between_matrices(m1, m2)
                    writer.add_histogram("pr2s/test_step", step_pr2s, global_step_val + batch_num)
                
            del rand, masked_tokens, noise, inputs_embeds, outputs
            
        # Calculate and log epoch-level metrics
        avg_loss = total_loss/len(test_dataloader)
        prediction_mean = total_prediction_mean/len(test_dataloader)
        label_mean = total_label_mean/len(test_dataloader)
        prediction_var = total_prediction_var/len(test_dataloader)
        label_var = total_label_var/len(test_dataloader)
        pr2 = total_pr2/len(test_dataloader)
        
        accelerator.print(f"Test_Loss: {avg_loss:.4f}, prediction_mean: {prediction_mean:.4f}, label_mean: {label_mean:.4f}, prediction_var: {prediction_var:.4f}, label_var: {label_var:.4f}, pr2: {pr2:.4f}\n")
        
        # Only keep the epoch-level logging with new suffixes
        writer.add_scalar("Loss/test_epoch", avg_loss, epoch)
        writer.add_scalar("Var/test_epoch/pred", prediction_var, epoch)
        writer.add_scalar("Var/test_epoch/label", label_var, epoch)
        writer.add_scalar("Mean/test_epoch/pred", prediction_mean, epoch)
        writer.add_scalar("Mean/test_epoch/label", label_mean, epoch)
        writer.add_scalar("pr2/test_epoch", pr2, epoch)

        # Log epoch-level histogram (only once)
        pr2s = BLM.plot_histogram_pr2s_between_matrices(m1, m2)
        writer.add_histogram("pr2s/test_epoch", pr2s, epoch)
        
def train_bacPT(args,ratio, fixed_mask, batch_size, save_dir):
    writer = SummaryWriter(args.output_path)
    LR = args.lr
    WARMUP = args.warmup
    HALF = args.half
    NO_CLIP = args.no_clip
    CLIP_VAL = args.clip_val
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(static_graph = True)])
    accelerator.print(args.output_path)
    if accelerator.is_main_process:
        seed_value = args.seed
        BLM.seed_everything(seed_value)
        print('~~~', seed_value)
    #TODO : Change dataset
    full_dataset = BLM.PCAGenomeDataset(num_samples=args.num_samples_total, max_seq_len = args.max_seq_len,\
                                            id_vs_num_filepath = '/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/dataset_final/train_proteins.csv',\
                                            parent_directory = "/orange/juannanzhou/bacteria_genome/protein_faa_files/sorted_esm_embeds/480_dim")
    train_size = int(ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size//2, shuffle=False)
    
    #TODO 
    #a) Hardcode the parameters of previous model, the cli only loads parameter for new model
    #b) The config of the new model should be defined below, however, the new model will be created from inside of a new function

    # Build longer model
    config = RoFormerConfig(
        hidden_size = args.hidden_size,
        max_position_embeddings = args.max_seq_len,
        num_hidden_layers = args.num_hidden_layers,
        num_attention_heads = args.num_attention_heads,
        attn_implementation = None,
        pca_dim = args.pca_dim,
        pca = args.pca,
        type_vocab_size = 1,
        hidden_dropout_prob = args.hidden_dropout_prob,
        hidden_act = args.hidden_act
    )
    
    model = extend_context_length(args.contig_model_ckpt_path, BLM.BacteriaLM_rope, config)
    # model = BLM.BacteriaLM_rope(config, add_pooling_layer = False)
    model.encoder.gradient_checkpointing = args.grad_check

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.10)

    # # Send everything through `accelerator.prepare`
    # train_dataloader, model, optimizer = accelerator.prepare(
    #     train_dataloader, model, optimizer
    # )
    
    if WARMUP is not None and args.scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP)
    elif WARMUP is not None and args.scheduler_type == "cosine":
        num_training_steps = (len(train_dataset)*args.epochs)//(args.batch_size*accelerator.state.num_processes)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,num_warmup_steps = WARMUP,num_training_steps = num_training_steps,num_cycles = args.num_cosine_cycles)
    
    scaler = None
    begin_epoch = 0
    if args.continue_training:
        # load model from checkpoint
        checkpoint_path = os.path.join(args.data_dir, 'checkpoints')
        last_epoch = BLM.get_last_epoch(checkpoint_path)
        print(last_epoch)
        checkpoint_path_epoch = os.path.join(checkpoint_path, 'model_' + str(last_epoch) + '.cp')
        checkpoint = torch.load(checkpoint_path_epoch, map_location='cpu')
        new_state_dict = OrderedDict((key.replace('module.', '',1), value) for key, value in checkpoint['model_state_dict'].items())
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        begin_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['train_loss']
        accelerator.print("loaded from checkpoint:"+str(checkpoint_path_epoch))
        accelerator.print("loaded from checkpoint, continuing from epoch "+str(begin_epoch))
    
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer
    )
    if HALF:
        accelerator.print("this is a mixed precision model")
        scaler = torch.cuda.amp.GradScaler()
    
    #test before any training happens
    if accelerator.is_main_process:
        writer.add_text('args', str(args), -1)
    if accelerator.is_main_process and not args.no_eval and not args.continue_training:
        evaluate_bacPT(full_dataset.ipca.mean_,full_dataset.ipca.var_,-1,writer,model, test_dataloader, 0.40, args.batch_size, accelerator, args.fixed_mask)
    
    epochs = args.epochs
    for epoch in range(begin_epoch, begin_epoch + epochs):
        if args.continue_training:
            #this is shoddy hack, and I hate this. keep begin_epoch as 0 to continue training
            mask_prob = cosine_mask_prob(epoch, 0, begin_epoch + epochs - 1, max_prob=0.4, min_prob=0.01)
        else:
            mask_prob = cosine_mask_prob(epoch, begin_epoch, begin_epoch + epochs - 1, max_prob=0.4, min_prob=0.01)
        # print('training started')
        model.train()
        start_time = time.time()
        loop = tqdm(train_dataloader, leave=True, disable=True)
        total_loss = 0
        total_prediction_mean = 0
        total_label_mean = 0
        total_prediction_var = 0
        total_label_var = 0
        total_pr2 = 0
        
        # Track global step for tensorboard logging
        global_step = epoch * len(train_dataloader)
        
        for batch_num, (inputs_embeds, labels, attention_mask) in enumerate(loop):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if fixed_mask:
                    seed_value = args.seed
                    set_seed(seed_value)
                rand = torch.rand(attention_mask.shape).to(accelerator.device)
                masked_tokens = (rand < mask_prob) & (attention_mask != 0)
                masked_tokens = torch.unsqueeze(masked_tokens, -1)
                if args.mask_type == 'random':
                    # random_tensor = torch.randn_like(inputs_embeds, dtype = torch.float32).to(accelerator.device)
                    mean = torch.zeros(full_dataset.ipca.mean_.shape).to(accelerator.device)
                    std_dev = torch.tensor(full_dataset.ipca.var_** 0.5).to(accelerator.device)
                    mean = mean.unsqueeze(0).unsqueeze(0).expand(inputs_embeds.size(0), inputs_embeds.size(1), -1)
                    std_dev = std_dev.unsqueeze(0).unsqueeze(0).expand_as(mean)
                    noise = torch.normal(mean = mean, std = std_dev).to(dtype=torch.float32)
                    inputs_embeds  = torch.where(masked_tokens, noise, inputs_embeds)
                if args.mask_type == 'noise':
                    # mean = torch.tensor(full_dataset.ipca.mean_).to(accelerator.device)
                    mean = torch.zeros(full_dataset.ipca.mean_.shape).to(accelerator.device)
                    std_dev = torch.tensor(full_dataset.ipca.var_** 0.5).to(accelerator.device)
                    mean = mean.unsqueeze(0).unsqueeze(0).expand(inputs_embeds.size(0), inputs_embeds.size(1), -1)
                    std_dev = std_dev.unsqueeze(0).unsqueeze(0).expand_as(mean)
                    noise = torch.normal(mean = mean, std = std_dev).to(dtype=torch.float32)
                    if args.noise_percentage_type == 'constant':
                        scaled_noise = noise * BLM.get_noise_percentage(args.noise_percentage_type, epoch, epochs, args.noise_percentage)
                    if args.noise_percentage_type == 'sine':
                        alpha, noise_r2 = BLM.sine_r2_to_alpha_scheduler(0.00, args.noise_max_r2, epochs, epoch, args.noise_num_cycles)
                        scaled_noise = noise * alpha
                    if args.noise_percentage_type == 'linear':
                        alpha, noise_r2 = BLM.linear_alpha_scheduler(args.noise_min_alpha, args.noise_max_alpha, epochs, epoch)
                        scaled_noise = noise * alpha
                    inputs_embeds = torch.where(masked_tokens, inputs_embeds + scaled_noise, inputs_embeds)
                if not NO_CLIP:
                    labels = torch.clip(labels,-CLIP_VAL,CLIP_VAL )
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = BLM.get_output_and_loss(model, inputs_embeds, attention_mask, masked_tokens, labels)
                        loss = outputs.loss
                        
                        # Update running totals (for epoch averages)
                        total_loss += loss.item()
                        total_prediction_mean += outputs.prediction_mean.item()
                        total_label_mean += outputs.label_mean.item()
                        total_prediction_var += outputs.prediction_var.item()
                        total_label_var += outputs.label_var.item()
                        total_pr2 += outputs.pr2.item()
                        
                        # Log metrics for each step (if main process)
                        if accelerator.is_main_process:
                            writer.add_scalar("Loss/train_step", loss.item(), global_step + batch_num)
                            writer.add_scalar("Var/train_step/pred", outputs.prediction_var.item(), global_step + batch_num)
                            writer.add_scalar("Var/train_step/label", outputs.label_var.item(), global_step + batch_num)
                            writer.add_scalar("Mean/train_step/pred", outputs.prediction_mean.item(), global_step + batch_num)
                            writer.add_scalar("Mean/train_step/label", outputs.label_mean.item(), global_step + batch_num)
                            writer.add_scalar("pr2/train_step", outputs.pr2.item(), global_step + batch_num)
                            
                        accelerator.backward(scaler.scale(loss))
                        accelerator.clip_grad_value_(model.parameters(), clip_value=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        m1, m2 = outputs.last_hidden_state_masked.detach().cpu().numpy(),\
                            outputs.ground_truth_masked.detach().cpu().numpy()
                if scheduler is not None:
                    scheduler.step()
                    last_lr = scheduler.get_last_lr()
                    last_lr = np.mean(last_lr)
                    # Log learning rate per step
                    if accelerator.is_main_process:
                        writer.add_scalar("learning_rate_step", last_lr, global_step + batch_num)
                
                # Log pr2 histogram for selected batches (e.g., every 10 batches)
                if accelerator.is_main_process and batch_num % 10 == 0:
                    pr2s = BLM.plot_histogram_pr2s_between_matrices(m1, m2)
                    writer.add_histogram("pr2s/train_step", pr2s, global_step + batch_num)
                    
                    # Log noise_r2 per step if applicable
                    if args.noise_percentage_type != 'constant' and args.mask_type != 'random':
                        writer.add_scalar("noise_r2_step", noise_r2, global_step + batch_num)
                
                del rand, masked_tokens, noise, inputs_embeds, outputs #delete outputs too
                torch.cuda.empty_cache()
                
        # Calculate epoch averages (keep this for epoch-level summary)
        avg_loss = total_loss/len(train_dataloader)
        prediction_mean = total_prediction_mean/len(train_dataloader)
        label_mean = total_label_mean/len(train_dataloader)
        prediction_var = total_prediction_var/len(train_dataloader)
        label_var =  total_label_var/len(train_dataloader)
        pr2 = total_pr2/len(train_dataloader)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1},Loss: {avg_loss:.4f}, prediction_mean: {prediction_mean:.4f}, label_mean: {label_mean:.4f}, prediction_var: {prediction_var:.4f}, label_var: {label_var:.4f}, pr2: {pr2:.4f}\n")
            
            # Still log epoch-level summaries
            writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
            writer.add_scalar("Var/train_epoch/pred", prediction_var, epoch)
            writer.add_scalar("Var/train_epoch/label", label_var, epoch)
            writer.add_scalar("Mean/train_epoch/pred", prediction_mean, epoch)
            writer.add_scalar("Mean/train_epoch/label", label_mean, epoch)
            writer.add_scalar("pr2/train_epoch", pr2, epoch)
            writer.add_scalar("learning_rate_epoch", last_lr, epoch)
            writer.add_scalar("mask_rate_epoch", mask_prob, epoch)
            
            # Log epoch-level histogram
            pr2s = BLM.plot_histogram_pr2s_between_matrices(m1,m2)
            writer.add_histogram("pr2s/train_epoch", pr2s, epoch)
            
            if args.noise_percentage_type != 'constant' and args.mask_type != 'random':
                writer.add_scalar("noise_r2_epoch", noise_r2, epoch)
        
        end_time = time.time()
        epoch_time = (end_time - start_time) / 60
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} took {epoch_time:.2f} minutes. \n")
        
        if epoch%args.eval_freq == 0 and accelerator.is_main_process and not args.no_eval:
            evaluate_bacPT(full_dataset.ipca.mean_,full_dataset.ipca.var_,epoch,writer,model, test_dataloader, mask_prob, args.batch_size, accelerator, args.fixed_mask)
        if epoch % args.checkpoint_freq == 0 and epoch != 0 and accelerator.is_main_process:
            checkpoint_path = os.path.join(args.data_dir, 'checkpoints')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_path_epoch = os.path.join(checkpoint_path, 'model_' + str(epoch) + '.cp')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
            }, checkpoint_path_epoch) 
    writer.flush()
    writer.close()
    
def main(args):
    train_bacPT(args,ratio = args.ratio_samples_train, fixed_mask = args.fixed_mask,\
                         batch_size = args.batch_size, \
                         save_dir = args.data_dir )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Trains the bacPT contig model")
    parser.add_argument('-num_samples_total', '--num_samples_total', type=int, help='Number of training samples', default=10)
    parser.add_argument('-ratio_samples_train', '--ratio_samples_train', type=float, help='Ratio of train samples', default=0.8)
    parser.add_argument('-d', '--data_dir', type=str, help='data directory', default="/orange/juannanzhou/bacteria_genome/roberta_bigger_runs/defaultattn_linear_alpha_allsamples/")
    parser.add_argument('-o', '--output_path', type=str, help='train output directory', default="/blue/juannanzhou/palash.sethi/Projects/bacteria_genome/data/noise_training/defaultattn_linear_alpha_allsamples")
    parser.add_argument('-cp', '--from_checkpoint', help='path to checkpoint if you want to re-run from checkpoint', default=None)
    parser.add_argument('--epochs', type=int, help='number epochs', default=10)
    parser.add_argument('-cp_freq', '--checkpoint_freq', type=int, help='checkpoint frequency (X epochs)', default=10)
    parser.add_argument('--eval_freq', type=int, help='frequency of evaluation (X epochs)', default=10)
    parser.add_argument('-lr', '--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--warmup', type=int, help='number of warm up steps', default=50)
    parser.add_argument('--scheduler_type', type=str, help='type of scheduler', default='constant')
    parser.add_argument('--num_cosine_cycles', type=int, help='number of cosine cycles', default=2)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=5)
    parser.add_argument('-m', '--max_seq_len', type=int, help='Max sequence length', default=50)
    parser.add_argument('-a', '--num_attention_heads', type=int, help='number of attention heads', default=10)
    parser.add_argument('-l', '--num_hidden_layers', type=int, help='number of hidden layers', default=12)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=480)
    parser.add_argument('--hidden_act', type=str, help='hidden activation', default='gelu')
    parser.add_argument('--hidden_dropout_prob', type=float, help='final mlp dropout prob', default= 0.1)
    # parser.add_argument('--pos_type', type=str, help="position embedding type", default="relative_key_query")
    parser.add_argument('--attn_type', type=str, help="attention type", default="flash_attention_2")
    # parser.add_argument('--return_dict_roberta', action='store_true', help='False if no hidden layers should be output', default=False)
    parser.add_argument('--no_eval', action='store_true', help='no evaluation', default=False)
    parser.add_argument('--half', action='store_true', help='half precision', default=True)
    parser.add_argument('--grad_check', action='store_true', help='gradient_checkpointing', default=True)
    parser.add_argument('--no_clip', action='store_true', help='no clipping of input embedding', default=True)
    parser.add_argument('--clip_val', type=int, help='clipping value (default -10,10)', default=10)
    # parser.add_argument('-mask_prob', '--mask_prob', type=float, help='mask probability', default=0.15)
    parser.add_argument('--mask_type', type=str, help="attention type", default="random")
    parser.add_argument('--noise_percentage_type', type=str, help="noise percentage type", default="constant")
    parser.add_argument('--noise_percentage', type=float, help='noise percentage to be added to input', default= 0.5)
    parser.add_argument('--noise_max_r2', type=float, help='max r2 for noise percentage', default= 0.2)
    parser.add_argument('--noise_num_cycles', type=float, help='number of noise cycles', default= 1)
    parser.add_argument('--noise_min_alpha', type=float, help='min alpha for noise percentage', default= 1.0)
    parser.add_argument('--noise_max_alpha', type=float, help='max alpha for noise percentage', default= 20.0)
    parser.add_argument('-fixed_mask', action='store_true', help='mask fixed positions only', default=False)
    parser.add_argument('-pca', action='store_true', help='regress on pca', default=False)
    parser.add_argument('-pca_dim', '--pca_dim', type=int, help='pca dimension', default=480)
    parser.add_argument('-seed', '--seed', type=int, help='seed value', default=25022024)
    parser.add_argument('-continue_training', action='store_true', help='continue from a previous checkpoint', default=False)
    parser.add_argument('--contig_model_ckpt_path', type=str, help='contig_model_ckpt_path', default="/orange/juannanzhou/bacteria_genome/final_checkpoints/RoformerContigFinal_v2")

    args = parser.parse_args()
    main(args)


