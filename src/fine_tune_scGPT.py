import argparse
import scanpy as sc
import anndata
import scgpt
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss
from scgpt.utils import load_pretrained
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
import wandb
import yaml
import copy
import os
import sys
import math


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # MATCH dtype to inputs (for AMP compatibility)
        alpha = self.alpha.to(inputs.dtype) if self.alpha is not None else None
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CrossEntropyLossWithAMP(nn.Module):
    """CrossEntropyLoss that handles AMP dtype conversion"""
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
    
    def forward(self, inputs, targets):
        # Convert weight to match input dtype (for AMP compatibility)
        weight = self.weight.to(inputs.dtype) if self.weight is not None else None
        return F.cross_entropy(inputs, targets, weight=weight)

def load_data(path_adata, device):
    '''
    Load fine-tune dataset
    '''
    adata = sc.read_h5ad(path_adata)
    return adata

def load_model(
    model_dir, 
    device, 
    num_classes=2, 
    num_batch_labels=None,
    freeze_encoder=False, 
    fast_transformer_backend="linear", 
    use_fast_transformer=True,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=False,
    do_mvc=True,
    use_mgm=False
):
    '''
    Load the pre-trained scGPT model with optional batch correction
    
    Args:
        do_dab: Enable Domain Adaptation via Reverse back-propagation (adversarial batch correction)
        use_batch_labels: Concatenate batch embeddings to decoder input (for batch-conditioned reconstruction)
                         NOTE: Keep False when loading pretrained models to avoid weight size mismatch!
                         Only set True if training from scratch or if checkpoint includes batch encoder.
        num_batch_labels: Number of unique batches (required if do_dab=True)
        domain_spec_batchnorm: Use domain-specific batch normalization
                              NOTE: Automatically enabled when do_dab=True (required to pass batch_labels)
        
    Important: do_dab requires either use_batch_labels OR domain_spec_batchnorm to pass batch_labels:
        - do_dab=True → Needs batch_labels for discriminator
        - We use domain_spec_batchnorm=True (no decoder size change, helps with batch correction)
        - use_batch_labels=True would also work but causes decoder size mismatch with pretrained weights
        
    For adversarial batch correction, you typically want:
        do_dab=True, use_batch_labels=False, domain_spec_batchnorm=True (auto-enabled)
    '''
    model_path = model_dir / "best_model.pt"
    vocab_path = model_dir / "vocab.json"
    config_path = model_dir / "args.json"

    # Load vocab
    vocab = GeneVocab.from_file(vocab_path)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    
    # Load model config
    with open(config_path, "r") as f:
        model_configs = json.load(f)

    # Validate batch-related parameters
    if do_dab and num_batch_labels is None:
        raise ValueError(
            "num_batch_labels must be specified when do_dab=True"
        )
    # In load_model function
    if use_mgm:
        # For MGM-only training, use n_cls=1 like pretraining
        n_cls_to_use = 1
    else:
        n_cls_to_use = num_classes
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Model n_cls configuration")
    print(f"{'='*80}")
    print(f"use_mgm = {use_mgm}")
    print(f"num_classes = {num_classes}")
    print(f"n_cls_to_use = {n_cls_to_use}")
    print(f"do_mvc = True (always for weight loading)")
    print(f"{'='*80}\n")

    # Create model with batch correction capabilities
    pad_token = "<pad>"
    pad_value = -2
    
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        n_cls=n_cls_to_use,
        vocab=vocab,
        dropout=model_configs.get("dropout", 0.2),
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=do_mvc,
        do_dab=do_dab,  # Enable adversarial batch correction
        use_batch_labels=use_batch_labels,  # Use batch labels in model
        num_batch_labels=num_batch_labels,  # Number of batches for discriminator
        domain_spec_batchnorm=domain_spec_batchnorm,  # Optional: domain-specific BN
        input_emb_style="continuous",
        n_input_bins=model_configs.get("n_bins", 51),
        cell_emb_style="cls",
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=False,
    )
    
    # Initialize model weights using checkpoint
    # Here I use load_pretrain instead of the loading function in the tutorial (https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Annotation.ipynb)
    # the reason is, that I have no installation of flash attention while the authors of scgpt have it. Using load_pretrained is just a wrapper around the code
    # used in the tutorial that safly maps parameters from flash attention layers back to normal attention layers. Thus it is safer to use the build in function here
    pretrained_dict = torch.load(model_path, map_location=device)
    try:
        load_pretrained(model, pretrained_dict, verbose=True)
        print("✓ Weights loaded with automatic conversion")
        print("\nChecking if MVC decoder loaded:")
        for name, param in model.named_parameters():
            if 'mvc_decoder' in name:
                print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    except Exception as e:
        print(f"⚠️  load_pretrained failed: {e}")
        print("Falling back to strict=False loading...")
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    
        if len(unexpected_keys) > 0:
            unexpected_non_discriminator = [k for k in unexpected_keys if 'grad_reverse_discriminator' not in k]
            if len(unexpected_non_discriminator) > 0:
                print(f"\n⚠️  Warning: Unexpected keys (non-discriminator):")
                for key in unexpected_non_discriminator[:5]:
                    print(f"    - {key}")
        
        if len(missing_keys) > 0:
            print(f"\n⚠️  Warning: Missing keys:")
            for key in missing_keys[:10]:
                print(f"    - {key}")

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name and "transformer_encoder" not in name:
                param.requires_grad = False
    
    model.to(device)
    return model, vocab, model_configs


def prepare_for_training(
    adata, 
    vocab, 
    key_label,
    class_assignments: dict,
    max_len,
    key_batch=None,
):
    '''
    Prepare data for training with optional batch labels
    
    Args:
        adata: AnnData object
        vocab: Gene vocabulary
        key_label: Column name for disease labels
        class_assignments: Maps dataset-specific disease labels to healthy/diseased
        key_batch: Column name for batch information (optional, required if using do_dab)
    
    Returns:
        data_dict: Dictionary with gene_ids, values, celltype_labels, and optionally batch_labels
    '''
    
    counts = adata.layers['X_binned']  # Log normalized binned counts expected in layers['X_binned']
    # only include genes in the pretrained model's vocab
    gene_ids = np.array([vocab[g] for g in adata.var['feature_name']])
    
    # Prepare disease labels
    labels = adata.obs[key_label].map({
        class_assignments['healthy']: 0, 
        class_assignments['diseased']: 1
    }).to_numpy()
    
    # Prepare batch labels if requested
    batch_labels = None
    if key_batch is not None:
        if key_batch not in adata.obs.columns:
            raise ValueError(
                f"Batch column '{key_batch}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        
        # Extract and encode batch IDs
        batch_ids_raw = adata.obs[key_batch].values
        unique_batches = np.unique(batch_ids_raw)
        batch_to_id = {batch: i for i, batch in enumerate(unique_batches)}
        batch_labels = np.array([batch_to_id[b] for b in batch_ids_raw])
    
    # Tokenize
    tokenized = tokenize_and_pad_batch(
        counts, gene_ids, max_len=max_len, vocab=vocab, 
        pad_token="<pad>", pad_value=-2, append_cls=True
    )
    
    # Create data dictionary
    data_dict = {
        "gene_ids": tokenized["genes"],
        "values": tokenized["values"],
        "celltype_labels": torch.from_numpy(labels).long()
    }
    
    # Add batch labels if provided
    if batch_labels is not None:
        data_dict["batch_labels"] = torch.from_numpy(batch_labels).long()
    
    return data_dict


def get_train_val_test_split(adata, key_donor='donor_id', key_label='disease', 
                              split_frac={'train': 0.7, 'val': 0.1, 'test': 0.2}):
    '''
    Stratified group split: splits donors into train/val/test while ensuring
    both disease classes are represented in each split.
    '''
    # Get one label per donor (assumes each donor has a single disease label)
    donor_labels = adata.obs.groupby(key_donor)[key_label].first()
    
    train_donors, val_donors, test_donors = [], [], []
    
    # Split each disease group of donors independently
    for label, group_donors in donor_labels.groupby(donor_labels):
        donors = group_donors.index.tolist()
        n = len(donors)
        n_train = max(1, round(n * split_frac['train']))
        n_val = max(1, round(n * split_frac['val']))
        
        rng = np.random.default_rng(42)
        donors = rng.permutation(donors).tolist()
        
        train_donors += donors[:n_train]
        val_donors   += donors[n_train:n_train + n_val]
        test_donors  += donors[n_train + n_val:]
    
    # Map donor sets back to cell indices
    donor_col = adata.obs[key_donor].values
    train_idx = np.where(np.isin(donor_col, train_donors))[0]
    val_idx   = np.where(np.isin(donor_col, val_donors))[0]
    test_idx  = np.where(np.isin(donor_col, test_donors))[0]
    
    return train_idx, val_idx, test_idx

def create_dataset_and_loader(data_dict, batch_size):
    dataset = SeqDataset(data_dict)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    vocab: GeneVocab,
    device: torch.device,
    epoch: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    gepc_weight: float,
    pad_token: str = "<pad>",
    amp: bool = True,
    log_interval: int = 100,
    # MGM parameters
    use_mgm: bool = False,
    use_gepc: bool = True,
    mgm_weight: float = 1.0,
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
    # Batch correction parameters
    do_dab: bool = False,
    dab_weight: float = 1.0,
    # Optimization parameters
    gradient_accumulation_steps: int = 1,
):
    """
    Training epoch with optional MGM and adversarial batch correction
    
    Args:
        do_dab: Enable adversarial batch correction
        dab_weight: Weight for batch adversarial loss (lambda parameter)
        gradient_accumulation_steps: Accumulate gradients over N batches (for memory efficiency)
    """
    model.train()
    
    # Use float for accumulation to avoid numerical issues
    total_loss = 0.0
    total_cls_loss = 0.0
    total_mgm_loss = 0.0
    total_gepc_loss = 0.0
    total_dab_loss = 0.0
    total_num = 0
    total_error = 0
    total_batch_error = 0  # For tracking batch discrimination accuracy
    
    # Zero gradients at start
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(loader):
        # Move data to device
        input_gene_ids = batch_data["gene_ids"].to(device, non_blocking=True)
        input_values = batch_data["values"].to(device, non_blocking=True)
        celltype_labels = batch_data["celltype_labels"].to(device, non_blocking=True)
        
        # Get batch labels if using batch correction
        batch_labels = None
        if do_dab:
            if "batch_labels" not in batch_data:
                raise ValueError(
                    "do_dab=True but batch_labels not found in data. "
                    "Make sure to set key_batch in config."
                )
            batch_labels = batch_data["batch_labels"].to(device, non_blocking=True)
        
        batch_size = input_gene_ids.size(0)
        
        # Store original values for MGM target (on GPU)
        original_values = input_values.clone()
        
        # Apply masking for MGM
        if use_mgm or use_gepc:
            masked_values = random_mask_value(
                values=input_values.cpu(),
                mask_ratio=mask_ratio,
                mask_value=mask_value,
                pad_value=pad_value
            ).to(device, non_blocking=True)
            
            masked_positions = (masked_values == mask_value) & (original_values != pad_value)
        else:
            masked_values = input_values
            masked_positions = None
        
        # Create padding mask
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                input_gene_ids,
                masked_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels,  # Pass batch labels to model
                CLS=not use_mgm,  # Only compute CLS if not doing MGM
                CCE=False,
                MVC=use_mgm or use_gepc,  # Enable MVC for MGM training OR if using GEPC auxiliary loss
                ECS=False,
            )
            if use_gepc:
                gepc_loss = torch.tensor(0.0, device=device)
            # Compute main task loss
            if use_mgm:
                # MGM loss (masked gene modeling)
                if "mlm_output" in output_dict:
                    mlm_output = output_dict["mlm_output"]
                    mgm_loss = masked_mse_loss(mlm_output, original_values, masked_positions)
                    loss = mgm_weight * mgm_loss
                    cls_loss = torch.tensor(0.0, device=device)
                else:
                    raise ValueError("MGM enabled but no mlm_output in model!")
            else:
                # Classification loss
                cls_output = output_dict["cls_output"]
                cls_loss = criterion(cls_output, celltype_labels)
                loss = cls_loss
                mgm_loss = torch.tensor(0.0, device=device)
            # GEPC loss (gene expression prediction from cell embeddings)
            gepc_loss = torch.tensor(0.0, device=device)
            if use_gepc:
                if "mvc_output" not in output_dict:
                    raise ValueError("use_gepc=True but no mvc_output in model!")
                mvc_output = output_dict["mvc_output"]
                gepc_loss = masked_mse_loss(mvc_output, original_values, masked_positions)
                loss = loss + gepc_weight * gepc_loss
            # Add adversarial batch correction loss
            dab_loss = torch.tensor(0.0, device=device)
            if do_dab:
                if "dab_output" not in output_dict:
                    raise ValueError(
                        "do_dab=True but dab_output not in model output. "
                        "This shouldn't happen - check model initialization."
                    )
                
                dab_output = output_dict["dab_output"]  # [batch_size, num_batches]
                dab_loss = F.cross_entropy(dab_output, batch_labels)
                
                # Add to total loss (gradient reversal handles the adversarial part!)
                loss = loss + dab_weight * dab_loss
        
        # Scale loss for gradient accumulation
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights (every N steps or at end of epoch)
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # Compute metrics WITHOUT gradients
        with torch.no_grad():
            if use_mgm:
                # MGM-only: no classification predictions
                predictions = None
                error = 0
            else:
                # Classification mode: compute predictions
                predictions = cls_output.argmax(dim=1)
                error = (predictions != celltype_labels).sum().item()
            total_error += error
            
            # Compute batch discrimination accuracy (to monitor if it's working)
            if do_dab:
                batch_predictions = dab_output.argmax(dim=1)
                batch_error = (batch_predictions != batch_labels).sum().item()
                total_batch_error += batch_error

        # Accumulate losses (detach to prevent memory leak)
        total_loss += loss.item() * batch_size * gradient_accumulation_steps
        total_cls_loss += cls_loss.item() * batch_size
        if use_mgm and isinstance(mgm_loss, torch.Tensor):
            total_mgm_loss += mgm_loss.item() * batch_size
        if use_gepc and isinstance(gepc_loss, torch.Tensor):
            total_gepc_loss += gepc_loss.item() * batch_size
        if do_dab and isinstance(dab_loss, torch.Tensor):
            total_dab_loss += dab_loss.item() * batch_size
        total_num += batch_size
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss_so_far = total_loss / total_num
            avg_cls_loss_so_far = total_cls_loss / total_num
            avg_mgm_loss_so_far = total_mgm_loss / total_num if use_mgm else 0.0
            avg_gepc_loss_so_far = total_gepc_loss / total_num if use_gepc else 0.0
            avg_dab_loss_so_far = total_dab_loss / total_num if do_dab else 0.0
            avg_error_so_far = total_error / total_num
            avg_batch_error_so_far = total_batch_error / total_num if do_dab else 0.0
            
            log_dict = {
                "train/batch_loss": avg_loss_so_far,
                "train/batch_cls_loss": avg_cls_loss_so_far,
                "train/batch_error": avg_error_so_far,
                "train/batch_idx": batch_idx,
                "epoch": epoch,
            }
            if use_mgm:
                log_dict["train/batch_mgm_loss"] = avg_mgm_loss_so_far
            if use_gepc:
                if "mvc_output" not in output_dict:
                    raise ValueError("use_gepc=True but no mvc_output in model!")
                mvc_output = output_dict["mvc_output"]
                log_dict["train/batch_gepc_loss"] = avg_gepc_loss_so_far

            if do_dab:
                log_dict["train/batch_dab_loss"] = avg_dab_loss_so_far
                log_dict["train/batch_discrimination_error"] = avg_batch_error_so_far
                log_dict["train/batch_discrimination_acc"] = 1.0 - avg_batch_error_so_far
            
            wandb.log(log_dict)
            
            # Console logging
            current_lr = optimizer.param_groups[0]['lr']
            log_msg = (
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {avg_loss_so_far:.4f} | CLS: {avg_cls_loss_so_far:.4f}"
            )
            if use_mgm:
                log_msg += f" | MGM: {avg_mgm_loss_so_far:.4f}"
            if use_gepc:  # ADD THIS
                log_msg += f" | GEPC: {avg_gepc_loss_so_far:.4f}"
            if do_dab:
                batch_acc = 1.0 - avg_batch_error_so_far
                log_msg += f" | DAB: {avg_dab_loss_so_far:.4f} (Batch Acc: {batch_acc:.4f})"
            log_msg += f" | LR: {current_lr:.2e}"
            print(log_msg)
        
        # Periodic memory cleanup
        if (batch_idx + 1) % 500 == 0:
            torch.cuda.empty_cache()
        
        # Clear variables to free memory
        del input_gene_ids, input_values, celltype_labels, original_values
        del masked_values, src_key_padding_mask, output_dict
        del loss, cls_loss, mgm_loss, dab_loss, predictions
        if batch_labels is not None:
            del batch_labels
        if masked_positions is not None:
            del masked_positions
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    # Return average metrics
    avg_loss = total_loss / total_num
    avg_cls_loss = total_cls_loss / total_num
    avg_mgm_loss = total_mgm_loss / total_num if use_mgm else 0.0
    avg_dab_loss = total_dab_loss / total_num if do_dab else 0.0
    avg_error = total_error / total_num
    avg_batch_error = total_batch_error / total_num if do_dab else 0.0
    
    return avg_loss, avg_cls_loss, avg_mgm_loss, avg_dab_loss, avg_error, avg_batch_error


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    vocab: GeneVocab,
    device: torch.device,
    pad_token: str = "<pad>",
    amp: bool = True,
    return_predictions: bool = False,
    # MGM parameters
    use_mgm: bool = False,
    use_gepc: bool = True,
    mgm_weight: float = 1.0,
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
    # Batch correction parameters
    do_dab: bool = False,
    dab_weight: float = 1.0,
):
    """Evaluate for one epoch with optional MGM and batch correction"""
    model.eval()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_mgm_loss = 0.0
    total_gepc_loss = 0.0
    total_dab_loss = 0.0
    total_num = 0
    total_error = 0
    total_batch_error = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            
            batch_labels = None
            if do_dab:
                if "batch_labels" not in batch_data:
                    raise ValueError("do_dab=True but batch_labels not found in data")
                batch_labels = batch_data["batch_labels"].to(device)
            
            batch_size = input_gene_ids.size(0)
            original_values = input_values.clone()
            
            if use_mgm or use_gepc:
                masked_values = random_mask_value(
                    values=input_values.cpu(),
                    mask_ratio=mask_ratio,
                    mask_value=mask_value,
                    pad_value=pad_value
                ).to(device)
                masked_positions = (masked_values == mask_value) & (original_values != pad_value)
            else:
                masked_values = input_values
                masked_positions = None
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    input_gene_ids,
                    masked_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels,
                    CLS=not use_mgm,
                    CCE=False,
                    MVC=use_mgm or use_gepc,
                    ECS=False,
                )
                if use_mgm:
                    if "mlm_output" in output_dict:
                        mlm_output = output_dict["mlm_output"]
                        mgm_loss = masked_mse_loss(mlm_output, original_values, masked_positions)
                        loss = mgm_weight * mgm_loss
                        cls_loss = torch.tensor(0.0, device=device)
                    else:
                        raise ValueError("MGM enabled but no mlm_output in model!")
                else:
                    cls_output = output_dict["cls_output"]
                    cls_loss = criterion(cls_output, celltype_labels)
                    loss = cls_loss
                    mgm_loss = torch.tensor(0.0, device=device)
                # GEPC loss (gene expression prediction from cell embeddings)
                gepc_loss = torch.tensor(0.0, device=device)
                if use_gepc:
                    if "mvc_output" not in output_dict:
                        raise ValueError("use_gepc=True but no mvc_output in model!")
                    mvc_output = output_dict["mvc_output"]
                    gepc_loss = masked_mse_loss(mvc_output, original_values, masked_positions)
                    loss = loss + gepc_loss
                
                dab_loss = torch.tensor(0.0, device=device)
                if do_dab:
                    dab_output = output_dict["dab_output"]
                    dab_loss = F.cross_entropy(dab_output, batch_labels)
                    loss = loss + dab_weight * dab_loss
            
            if use_mgm:
                predictions = None
                error = 0
            else:
                predictions = cls_output.argmax(dim=1)
                error = (predictions != celltype_labels).sum().item()
                
                if return_predictions:
                    probs = F.softmax(cls_output, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(celltype_labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            if do_dab:
                batch_predictions = dab_output.argmax(dim=1)
                batch_error = (batch_predictions != batch_labels).sum().item()
                total_batch_error += batch_error
            
            total_error += error
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            if use_mgm:
                total_mgm_loss += mgm_loss.item() * batch_size
            if do_dab:
                total_dab_loss += dab_loss.item() * batch_size
            total_num += batch_size
    
    avg_loss = total_loss / total_num
    avg_cls_loss = total_cls_loss / total_num
    avg_mgm_loss = total_mgm_loss / total_num if use_mgm else 0.0
    avg_dab_loss = total_dab_loss / total_num if do_dab else 0.0
    avg_error = total_error / total_num
    avg_batch_error = total_batch_error / total_num if do_dab else 0.0
    
    if return_predictions:
        return (avg_loss, avg_cls_loss, avg_mgm_loss, avg_dab_loss, avg_error, avg_batch_error,
                np.array(all_predictions), np.array(all_labels), np.array(all_probs))
    else:
        return avg_loss, avg_cls_loss, avg_mgm_loss, avg_dab_loss, avg_error, avg_batch_error


def compute_metrics(labels, predictions, probs, class_assignments):
    """Compute classification metrics"""
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0, average='macro'),
    }
    
    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc'] = 0.0
    
    # Add per-class metrics
    class_names = [class_assignments['healthy'], class_assignments['diseased']]
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    # Add confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
    
    return metrics

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                     num_cycles=0.5, min_lr_ratio=0.1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer. Cosine schedule that doesn't go below
      min_lr_ratio * initial_lr
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of waves in the cosine schedule (default is 0.5).
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        
        # Don't go below min_lr_ratio (default 10% of initial lr)
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)

def train_scgpt_classifier(
    model,
    criterion,
    device,
    epochs,
    train_loader,
    val_loader,
    use_gepc,
    vocab,
    config,
    logger,
    save_dir,
    class_assignments
):
    """Main training function"""
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    # Calculate total training steps
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1  # Never go below 1e-5 (10% of 1e-4)
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config['training'].get('amp', True))
    
    best_val_loss = float('inf')
    best_macro_f1 = 0.0
    best_model = None
    class_names = [class_assignments['healthy'], class_assignments['diseased']]
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    # Extract training parameters
    use_mgm = config['training'].get('use_mgm', False)
    do_dab = config['model'].get('do_dab', False)
    
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Training objective: {'MGM (Masked Gene Modeling)' if use_mgm else 'Classification'}")
    logger.info(f"Adversarial batch correction: {'Enabled' if do_dab else 'Disabled'}")
    if do_dab:
        logger.info(f"  DAB weight (λ): {config['training'].get('dab_weight', 1.0)}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['lr']}")
    logger.info(f"{'='*80}\n")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_loss, train_cls_loss, train_mgm_loss, train_dab_loss, train_error, train_batch_error = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            vocab=vocab,
            device=device,
            gepc_weight=config['training'].get('gepc_weight', 1.0),
            epoch=epoch,
            amp=config['training'].get('amp', True),
            use_mgm=use_mgm,
            use_gepc=use_gepc, # scGPT authors recommend using GEPC loss to improve cell embedding quality
            mgm_weight=config['training'].get('mgm_weight', 1.0),
            mask_ratio=config['training'].get('mask_ratio', 0.15),
            mask_value=config['training'].get('mask_value', -1),
            pad_value=config['training'].get('pad_value', -2),
            do_dab=do_dab,
            dab_weight=config['training'].get('dab_weight', 1.0),
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        )
        
        # Validation
        val_loss, val_cls_loss, val_mgm_loss, val_dab_loss, val_error, val_batch_error, val_preds, val_labels, val_probs = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            vocab=vocab,
            device=device,
            amp=config['training'].get('amp', True),
            return_predictions=True,
            use_mgm=use_mgm,
            use_gepc=config['training'].get('use_gepc', True),
            mgm_weight=config['training'].get('mgm_weight', 1.0),
            mask_ratio=config['training'].get('mask_ratio', 0.15),
            mask_value=config['training'].get('mask_value', -1),
            pad_value=config['training'].get('pad_value', -2),
            do_dab=do_dab,
            dab_weight=config['training'].get('dab_weight', 1.0),
        )
        
        # Compute validation metrics
        val_metrics = compute_metrics(val_labels, val_preds, val_probs, class_assignments) if not use_mgm else {}
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/cls_loss": train_cls_loss,
            "train/error": train_error,
            "val/loss": val_loss,
            "val/cls_loss": val_cls_loss,
            "val/error": val_error,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time,
        }
        
        if use_mgm:
            log_dict.update({
                "train/mgm_loss": train_mgm_loss,
                "val/mgm_loss": val_mgm_loss,
            })
        
        if do_dab:
            log_dict.update({
                "train/dab_loss": train_dab_loss,
                "train/batch_discrimination_acc": 1.0 - train_batch_error,
                "val/dab_loss": val_dab_loss,
                "val/batch_discrimination_acc": 1.0 - val_batch_error,
            })
        
        if not use_mgm:
            log_dict.update({
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
                "val/auc": val_metrics['auc'],
            })
        
        wandb.log(log_dict)
        
        # Console output
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch}/{epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"{'='*80}")
        logger.info(f"Train - Loss: {train_loss:.4f} | CLS: {train_cls_loss:.4f} | Error: {train_error:.4f}")
        if use_mgm:
            logger.info(f"      - MGM: {train_mgm_loss:.4f}")
        if do_dab:
            logger.info(f"      - DAB: {train_dab_loss:.4f} | Batch Acc: {1.0-train_batch_error:.4f}")
        
        logger.info(f"Val   - Loss: {val_loss:.4f} | CLS: {val_cls_loss:.4f} | Error: {val_error:.4f}")
        if use_mgm:
            logger.info(f"      - MGM: {val_mgm_loss:.4f}")
        if do_dab:
            logger.info(f"      - DAB: {val_dab_loss:.4f} | Batch Acc: {1.0-val_batch_error:.4f}")
        
        if not use_mgm:
            logger.info(f"      - Acc: {val_metrics['accuracy']:.4f} | "
                       f"Prec: {val_metrics['precision']:.4f} | "
                       f"Rec: {val_metrics['recall']:.4f} | "
                       f"F1: {val_metrics['f1']:.4f} | "
                       f"AUC: {val_metrics['auc']:.4f}")
                        # Per-class metrics
            logger.info(f"      - Per-Class Metrics:")
            for class_name in class_names:
                logger.info(f"        {class_name:>10s}: "
                           f"Prec={val_metrics[f'precision_{class_name}']:.4f} | "
                           f"Rec={val_metrics[f'recall_{class_name}']:.4f} | "
                           f"F1={val_metrics[f'f1_{class_name}']:.4f}")
            
            # Confusion Matrix
            cm = val_metrics['confusion_matrix']
            logger.info(f"      - Confusion Matrix:")
            logger.info(f"                    Predicted")
            logger.info(f"               {class_names[0]:<10s} {class_names[1]:<10s}")
            logger.info(f"        Actual")
            for i, class_name in enumerate(class_names):
                logger.info(f"        {class_name:<10s} {cm[i, 0]:<10d} {cm[i, 1]:<10d}")
        logger.info(f"{'='*80}\n")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if not use_mgm:
            history['val_metrics'].append(val_metrics)
        
        # Early stopping for mgm using val loss
        if config['training']['use_mgm']:
            # early_stopping_metric = best_macro_f1
            # best_early_stopping_metric = val_metrics['f1']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
                torch.save(best_model, save_dir / "best_model.pt")
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{config['training']['patience']})")
                
                if patience_counter >= config['training']['patience']:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        # early stopping for BCE using macro F1
        else:
            macro_f1 = val_metrics['f1']
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
                torch.save(best_model, save_dir / "best_model.pt")
                logger.info(f"✓ New best model saved (macro F1: {macro_f1:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{config['training']['patience']})")
                
                if patience_counter >= config['training']['patience']:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    break
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--sweep', action='store_true', help='Running as wandb sweep')
    args = parser.parse_args()
    
    is_sweep = args.sweep
    
    if is_sweep:
        # Wandb sweep initialization
        run = wandb.init()
        
        if not args.config:
            raise ValueError("--config is required even for sweep mode (provides base config)")
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config with sweep parameters
        print(f"\n{'='*80}")
        print("WANDB SWEEP - Overriding Config Parameters")
        print(f"{'='*80}")
        for key, value in wandb.config.items():
            if '.' in key:
                section, param = key.split('.', 1)
                if section not in config:
                    config[section] = {}
                
                old_value = config[section].get(param, 'not set')
                updated = old_value != value
                config[section][param] = value
                
                if updated:
                    print(f"  {section}.{param}: {old_value} → {value}")
            else:
                old_value = config['training'].get(key, 'not set')
                updated = old_value != value
                config['training'][key] = value
                
                if updated:
                    print(f"  training.{key}: {old_value} → {value}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("RUNNING IN NORMAL MODE")
        print(f"{'='*80}\n")
        
        if not args.config:
            raise ValueError("--config is required for normal mode")
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        run_name = config['wandb'].get('run_name', f"scFM_{time.strftime('%Y%m%d_%H%M%S')}")
        
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config
        )
        
        print(f"✓ Wandb initialized successfully!")
        print(f"  Run name: {wandb.run.name}")
        print(f"  Project: {wandb.run.project}")
        print(f"  URL: {wandb.run.url}\n")
    
    # Create save directory
    adata_path = Path(config['data']['adata_path'])
    model_dir = Path(config['model']['model_dir'])
    save_dir = Path(config['model']['save_dir'])
    if is_sweep:
        save_dir = save_dir / f"sweep_{wandb.run.id}"
        print(f"Sweep: Creating unique save directory")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(f"cuda:{config['model']['device']}")
    print(f"\nDevice Configuration:")
    print(f"  Device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(config['model']['device'])}")
    print(f"  Memory: {torch.cuda.get_device_properties(config['model']['device']).total_memory / 1e9:.2f} GB\n")
    
    # Load data
    print("Loading data...")
    adata = load_data(adata_path, device=device)
    
    # Determine batch correction settings
    do_dab = config['model'].get('do_dab', False)
    use_gepc = config['training'].get('use_gepc', True)
    use_batch_labels = config['model'].get('use_batch_labels', False)  # Keep as False by default
    max_len = config['model']['max_len']
    
    # When using do_dab, we need to pass batch_labels to the model
    # This requires either use_batch_labels=True OR domain_spec_batchnorm=True
    # We use domain_spec_batchnorm=True by default (doesn't change decoder size)
    domain_spec_batchnorm = config['model'].get('domain_spec_batchnorm', do_dab)  # Auto-enable with do_dab
    
    key_batch = config['data'].get('key_batch', None)
    
    if do_dab and key_batch is None:
        raise ValueError(
            "do_dab=True but key_batch not specified in config. "
            "Please add 'key_batch' to the data section of your config file."
        )
    
    # Count unique batches if using batch correction
    num_batch_labels = None
    if do_dab:
        if key_batch not in adata.obs.columns:
            raise ValueError(
                f"Batch column '{key_batch}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        num_batch_labels = len(adata.obs[key_batch].unique())
        print(f"\nBatch correction enabled:")
        print(f"  Batch column: {key_batch}")
        print(f"  Number of batches: {num_batch_labels}")
    use_mgm = config['training'].get('use_mgm', False)
    # Load model
    print("\nLoading model...")
    model, vocab, pretrain_config = load_model(
        model_dir=model_dir,
        device=device,
        num_classes=config['model'].get('num_classes', 2),
        num_batch_labels=num_batch_labels,
        freeze_encoder=config['model']['freeze_encoder'],
        fast_transformer_backend=config['model']['fast_transformer_backend'],
        use_fast_transformer=config['model']['use_fast_transformer'],
        do_dab=do_dab,
        use_batch_labels=use_batch_labels,
        domain_spec_batchnorm=domain_spec_batchnorm,
        do_mvc=config['model'].get('do_mvc', True),
        use_mgm=use_mgm
    )
    
    # Verify gene overlap
    if 'feature_name' not in adata.var:
        adata.var['feature_name'] = adata.var.index.values
    matched_genes = sum(g in vocab for g in adata.var['feature_name'])
    print(f"Matched {matched_genes}/{len(adata.var['feature_name'])} genes in vocab")
    
    # Prepare data splits
    print("\nPreparing train/val/test splits...")
    class_assignments = {'healthy': config['data']['healthy_class'],
                         'diseased': config['data']['diseased_class']}
    
    train_idx, val_idx, test_idx = get_train_val_test_split(
        adata, 
        key_donor=config['data']['key_donor']
    )
    
    data_train = prepare_for_training(
        adata[train_idx], 
        vocab=vocab, 
        key_label=config['data']['key_label'], 
        class_assignments=class_assignments,
        key_batch=key_batch,
        max_len=max_len
    )
    data_val = prepare_for_training(
        adata[val_idx], 
        vocab=vocab, 
        key_label=config['data']['key_label'], 
        class_assignments=class_assignments,
        key_batch=key_batch,
        max_len=max_len
    )
    data_test = prepare_for_training(
        adata[test_idx], 
        vocab=vocab, 
        key_label=config['data']['key_label'], 
        class_assignments=class_assignments,
        key_batch=key_batch,
        max_len=max_len
    )
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = create_dataset_and_loader(data_train, batch_size=batch_size)
    val_loader = create_dataset_and_loader(data_val, batch_size=batch_size)
    test_loader = create_dataset_and_loader(data_test, batch_size=batch_size)
    
    # Setup loss function
    use_mgm = config["training"].get("use_mgm", False)
    
    if use_mgm:
        criterion = None
    else:
        if config["training"].get("use_class_weights", False):
            train_labels_np = data_train['celltype_labels'].numpy()
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels_np),
                y=train_labels_np
            )
            
            class_weight_boost = config["training"].get("class_weight_boost", 1.0)
            if class_weight_boost != 1.0:
                class_weights[0] = class_weights[0] / class_weight_boost
            
            print(f"\n{'='*80}")
            print("CLASS IMBALANCE HANDLING")
            print(f"{'='*80}")
            print(f"Training set distribution:")
            print(f"  Class 0 ({config['data']['healthy_class']}):    {np.sum(train_labels_np == 0):>6} samples")
            print(f"  Class 1 ({config['data']['diseased_class']}): {np.sum(train_labels_np == 1):>6} samples")
            print(f"  Imbalance ratio:     1:{np.sum(train_labels_np == 1)/np.sum(train_labels_np == 0):.2f}")
            print(f"\nComputed class weights (boost={class_weight_boost}):")
            print(f"  Class 0 weight: {class_weights[0]:.4f}")
            print(f"  Class 1 weight: {class_weights[1]:.4f}")
            print(f"{'='*80}\n")
            
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)
            if config['training'].get('gamma'):
                gamma = config['training']['gamma']
                criterion = FocalLoss(alpha=class_weights_tensor, gamma=gamma)
            else:
                criterion = CrossEntropyLossWithAMP(weight=class_weights_tensor)
        else:
            if config['training'].get('gamma'):
                gamma = config['training']['gamma']
                criterion = FocalLoss(gamma=gamma)
            else:
                criterion = CrossEntropyLossWithAMP()
    
    # Train model
    logger = scgpt.logger
    
    best_model, history = train_scgpt_classifier(
        model=model,
        criterion=criterion,
        device=device,
        epochs=config['training']['epochs'],
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        use_gepc=use_gepc,
        config=config,
        logger=logger,
        save_dir=save_dir,
        class_assignments=class_assignments
    )
    
    # Final test evaluation
    print("\nRunning final test evaluation...")
    
    test_results = evaluate_epoch(
        best_model, 
        test_loader, 
        criterion, 
        vocab, 
        device,
        amp=config["training"].get("amp", True),
        return_predictions=True,
        use_mgm=config["training"].get("use_mgm", False),
        use_gepc=use_gepc,
        mgm_weight=config["training"].get("mgm_weight", 1.0),
        mask_ratio=config["training"].get("mask_ratio", 0.15),
        mask_value=config["training"].get("mask_value", -1),
        pad_value=config["training"].get("pad_value", -2),
        do_dab=do_dab,
        dab_weight=config["training"].get("dab_weight", 1.0),
    )
    
    test_loss, test_cls_loss, test_mgm_loss, test_dab_loss, test_error, test_batch_error = test_results[:6]
    
    if not use_mgm:
        test_preds, test_labels, test_probs = test_results[6:]
        test_metrics = compute_metrics(test_labels, test_preds, test_probs, class_assignments)
    else:
        test_metrics = {}
    
    # Log final results
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST RESULTS:")
    logger.info(f"  Total Loss: {test_loss:.4f}")
    logger.info(f"  CLS Loss:   {test_cls_loss:.4f}")
    if use_mgm:
        logger.info(f"  MGM Loss:   {test_mgm_loss:.4f}")
    if do_dab:
        logger.info(f"  DAB Loss:   {test_dab_loss:.4f}")
        logger.info(f"  Batch Disc Acc: {1.0-test_batch_error:.4f}")
    logger.info(f"  Error Rate: {test_error:.4f}")
    if not use_mgm:
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
    logger.info("="*80)
    
    # Log to wandb
    wandb_test_log = {
        "test/loss": test_loss,
        "test/cls_loss": test_cls_loss,
    }
    if use_mgm:
        wandb_test_log["test/mgm_loss"] = test_mgm_loss
    if do_dab:
        wandb_test_log["test/dab_loss"] = test_dab_loss
        wandb_test_log["test/batch_discrimination_acc"] = 1.0 - test_batch_error
    if not use_mgm:
        wandb_test_log.update({
            "test/accuracy": test_metrics['accuracy'],
            "test/f1": test_metrics['f1'],
            "test/auc": test_metrics['auc'],
        })
    
    wandb.log(wandb_test_log)
    wandb.finish()
    
    return best_model, history


if __name__ == "__main__":
    main()