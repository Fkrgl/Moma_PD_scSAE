"""
Extract Cell Embeddings from scGPT Models (Pretrained or Fine-tuned)

This script:
1. Loads pretrained or fine-tuned scGPT models (.pt file)
2. Auto-detects batch correction (DSBN, DAB)
3. Extracts cell embeddings using scGPT's built-in function
4. Saves embeddings to AnnData

FEATURES:
- Works with pretrained models (zero-shot)
- Works with fine-tuned models (with or without batch correction)
- Auto-detects model architecture from checkpoint
- Handles flash-attn vs standard transformer weights
- Uses L2-normalized embeddings (consistent with scGPT)

UPDATED: 2026-02-04 - Now uses get_batch_cell_embeddings for robust extraction
"""

import argparse
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Optional
import json
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import os
from scipy.sparse import issparse
import scib

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tasks import get_batch_cell_embeddings
from scgpt.utils import load_pretrained


def load_model(
    pretrained_model_dir: Path,
    finetuned_weights_path: Path,
    device: str = "cuda",
    num_classes: int = 2,
    with_batch_correction: bool = None,
    num_batch_labels: Optional[int] = None
):
    """
    Load scGPT model with automatic architecture detection. The function uses scGPTs load_pretrained
    to load the weights of the model. This is important to not missmatch the pre-training architect-
    ure build on flash attention
    
    Args:
        pretrained_model_dir: Directory with pretrained model (vocab.json, args.json)
        finetuned_weights_path: Path to weights file (.pt)
        device: Device to load model on
        num_classes: Number of classes for fine-tuned models (ignored for pretrained)
        with_batch_correction: Whether model has batch correction (None = auto-detect)
        num_batch_labels: Number of batches (None = auto-detect from weights)
    
    Returns:
        model, vocab, config, with_batch_correction
    """
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    
    # Load vocabulary
    vocab_path = pretrained_model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_path)
    
    # Ensure special tokens exist
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    
    print(f"✓ Loaded vocabulary: {len(vocab)} genes")
    
    # Load model config
    config_path = pretrained_model_dir / "args.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Loaded config:")
    print(f"  - Embedding size: {config['embsize']}")
    print(f"  - Layers: {config['nlayers']}")
    print(f"  - Heads: {config['nheads']}")
    
    # Load state dict to inspect it
    print(f"\n✓ Loading weights from: {finetuned_weights_path}")
    state_dict = torch.load(finetuned_weights_path, map_location=device)
    
    # Auto-detect model type from state_dict
    has_classifier = any('cls_decoder' in key for key in state_dict.keys())
    has_mvc = any('mvc_decoder' in key for key in state_dict.keys())
    has_dsbn = any('dsbn' in key for key in state_dict.keys())
    has_discriminator = any('grad_reverse_discriminator' in key for key in state_dict.keys())
    
    # Determine architecture parameters
    if has_classifier:
        # Fine-tuned model - detect n_cls from classifier weights
        print("\n✓ Detected: Fine-tuned model")
        cls_keys = [k for k in state_dict.keys() if 'cls_decoder' in k and 'weight' in k]
        if cls_keys:
            if 'cls_decoder.out_layer.weight' in state_dict:
                n_cls = state_dict['cls_decoder.out_layer.weight'].shape[0]
            print(f"  - Number of classes: {n_cls}")
        else:
            n_cls = num_classes
        nlayers_cls = 3
        do_mvc = has_mvc
    else:
        # Pretrained model
        print("\n✓ Detected: Pretrained model")
        nlayers_cls = config.get("n_layers_cls", 3)
        n_cls = 1
        do_mvc = True  # Pretrained always has MVC
    
    # Detect batch correction
    if has_dsbn or has_discriminator:
        print(f"\n⚠️  Detected batch correction in saved model:")
        if has_dsbn:
            dsbn_keys = [k for k in state_dict.keys() if k.startswith('dsbn.bns.') and 'running_mean' in k]
            detected_num_batches = len(dsbn_keys)
            print(f"  - DSBN layers found: {detected_num_batches} batches")
            if num_batch_labels is None:
                num_batch_labels = detected_num_batches
                print(f"  - Auto-detected num_batch_labels: {num_batch_labels}")
        if has_discriminator:
            print(f"  - Adversarial discriminator found")
        
        if with_batch_correction is None:
            with_batch_correction = True
            print(f"  - Auto-setting with_batch_correction=True")
    elif with_batch_correction is None:
        with_batch_correction = False
    
    # Initialize model with detected architecture
    print(f"\nInitializing model...")
    print(f"  - Model type: {'Fine-tuned' if has_classifier else 'Pretrained'}")
    print(f"  - Classification layers: {nlayers_cls}")
    print(f"  - Number of classes: {n_cls}")
    print(f"  - MVC decoder: {do_mvc}")
    print(f"  - Batch correction: {with_batch_correction}")
    if with_batch_correction:
        print(f"  - Number of batches: {num_batch_labels}")
    
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config['embsize'],
        nhead=config['nheads'],
        d_hid=config['d_hid'],
        nlayers=config['nlayers'],
        nlayers_cls=nlayers_cls,
        n_cls=n_cls,
        vocab=vocab,
        dropout=config['dropout'],
        pad_token=config['pad_token'],
        pad_value=config['pad_value'],
        do_mvc=do_mvc,
        # Batch correction parameters
        do_dab=with_batch_correction,
        use_batch_labels=False,
        num_batch_labels=num_batch_labels if with_batch_correction else None,
        domain_spec_batchnorm=with_batch_correction,
        explicit_zero_prob=config.get('explicit_zero_prob', False),
        use_fast_transformer=False,  # Use standard transformer for compatibility
        fast_transformer_backend='linear',
        pre_norm=config.get('pre_norm', False)
    )
    
    #Load weights with automatic conversion
    try:
        load_pretrained(model, state_dict, verbose=False)
        print("\n✓ Weights loaded with automatic conversion")
    except Exception as e:
        print(f"\n⚠️  load_pretrained failed: {e}")
        print("Falling back to strict=False loading...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
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
    
    model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded successfully on {device}")
    if with_batch_correction:
        print(f"  Note: Discriminator weights ignored (not needed for inference)")
    print(f"{'='*80}\n")
    
    return model, vocab, config, with_batch_correction


def extract_embeddings(
    adata, 
    pretrained_model_dir, 
    finetuned_weights_path, 
    with_batch_correction=None,
    batch_col="sample",
    device="cuda:2",
    batch_size=128,
    max_len=3001,
    num_classes=2
):
    """
    Fast embedding extraction that properly handles batch labels.
    Uses the same efficient DataLoader approach as get_batch_cell_embeddings() but it calls model._encode() directly.
    This allows for maximal control. For mor information read: scGPT_embedding_extraction_summary.md (local machine)
    """
    
    # Load model
    model, vocab, config, with_batch_correction = load_model(
        Path(pretrained_model_dir), 
        Path(finetuned_weights_path),
        device=device, 
        num_classes=num_classes, 
        with_batch_correction=with_batch_correction, 
        num_batch_labels=None
    )
    
    # Filter to genes in vocabulary
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 
        for gene in adata.var['feature_name']
    ]
    
    print(f"\nGene filtering:")
    print(f"  Before: {adata.shape[1]} genes")
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    print(f"  After: {adata.shape[1]} genes")

    lengths = []
    for i in range(adata.shape[0]):
        row = adata.X[i].toarray() if issparse(adata.X) else adata.X[i]
        nonzero_count = np.count_nonzero(row)
        lengths.append(nonzero_count + 1)  # +1 for <cls> token

    print(f"Max sequence length: {max(lengths)}")
    print(f"Number of cells > 3001: {sum(l > 3001 for l in lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Median length: {np.median(lengths):.1f}")
    
    # Get count matrix
    count_matrix = adata.layers['X_binned'] # use binned expression values as input
    count_matrix = count_matrix.toarray() if issparse(count_matrix) else count_matrix
    
    # Get gene IDs
    gene_ids = np.array(adata.var["id_in_vocab"])
    
    # Prepare batch labels if needed
    if with_batch_correction:
        print(f"\n✓ Model has batch correction")
        if batch_col not in adata.obs.columns:
            raise ValueError(f"Batch column '{batch_col}' not found")
        
        batch_ids_raw = adata.obs[batch_col].values
        unique_batches = sorted(np.unique(batch_ids_raw))
        batch_to_id = {batch: i for i, batch in enumerate(unique_batches)}
        batch_ids = np.array([batch_to_id[b] for b in batch_ids_raw])
        
        print(f"  Found {len(unique_batches)} unique batches")
    else:
        print(f"\n✓ Model has no batch correction")
        batch_ids = None
    
    # Custom Dataset (same as in get_batch_cell_embeddings)
    class CellDataset(Dataset):
        def __init__(self, count_matrix, gene_ids, batch_ids=None):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids
            self.batch_ids = batch_ids
        
        def __len__(self):
            return len(self.count_matrix)
        
        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            
            # Append <cls> token at the beginning
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, config["pad_value"])
            
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            
            output = {
                "genes": genes,
                "expressions": values,
            }
            
            if self.batch_ids is not None:
                output["batch_labels"] = self.batch_ids[idx]
            
            return output
    
    # Custom collator that preserves batch_labels
    def collate_fn(batch):
        """Custom collator that handles batch_labels"""
        # Get max length in batch
        max_len_batch = min(max(len(x["genes"]) for x in batch), max_len)
        
        # Prepare padded tensors
        genes_padded = []
        exprs_padded = []
        batch_labels_list = []
        
        for item in batch:
            genes = item["genes"]
            exprs = item["expressions"]
            
            # Truncate or pad
            if len(genes) > max_len_batch:
                genes = genes[:max_len_batch]
                exprs = exprs[:max_len_batch]
            else:
                pad_len = max_len_batch - len(genes)
                genes = torch.cat([genes, torch.full((pad_len,), vocab[config["pad_token"]], dtype=torch.long)])
                exprs = torch.cat([exprs, torch.full((pad_len,), config["pad_value"], dtype=torch.float)])
            
            genes_padded.append(genes)
            exprs_padded.append(exprs)
            
            if "batch_labels" in item:
                batch_labels_list.append(item["batch_labels"])
        
        result = {
            "gene": torch.stack(genes_padded),
            "expr": torch.stack(exprs_padded),
        }
        
        if batch_labels_list:
            result["batch_labels"] = torch.tensor(batch_labels_list, dtype=torch.long)
        
        return result
    
    # Create dataset and dataloader (same efficient approach)
    dataset = CellDataset(count_matrix, gene_ids, batch_ids if with_batch_correction else None)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size // 2),  # Multi-worker!
        pin_memory=True,
    )
    
    # Extract embeddings (same as get_batch_cell_embeddings)
    print(f"\nExtracting embeddings...")
    model.eval()
    
    cell_embeddings = np.zeros((len(dataset), config["embsize"]), dtype=np.float32)
    
    from tqdm import tqdm
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells"):
            input_gene_ids = data_dict["gene"].to(device)
            input_values = data_dict["expr"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[config["pad_token"]])  # masking is only used on the padding token, expreesion values must been unmasked for embeddinge extraction!
            
            # Get batch labels if present
            batch_labels_batch = None
            if "batch_labels" in data_dict:
                batch_labels_batch = data_dict["batch_labels"].to(device)
            
            # Encode
            embeddings = model._encode(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels_batch,
            )
            
            embeddings = embeddings[:, 0, :]  # Get <cls> token embedding
            embeddings = embeddings.cpu().numpy()
            
            cell_embeddings[count : count + len(embeddings)] = embeddings
            count += len(embeddings)
    
    # L2 normalization
    print("no L2 normalization!")
    # cell_embeddings = cell_embeddings / np.linalg.norm(
    #     cell_embeddings, axis=1, keepdims=True
    # )
    
    print(f"  Shape: {cell_embeddings.shape}")
    print(f"  Stats: mean={cell_embeddings.mean():.6f}, std={cell_embeddings.std():.6f}")
    
    #adata.obsm["X_scGPT"] = cell_embeddings
    print(f"\n✓ Done!")
    
    return cell_embeddings

def get_embedding_quality_metrics(adata, batch_key, celltype_key, disease_key, embed="X_scGPT", n_cores=5):
    # we need to compute kNN graph on our extracted embedding, tahts what scbi operates on
    sc.pp.neighbors(adata, use_rep="X_scGPT", n_neighbors=15)
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=celltype_key,
        embed=embed,
        isolated_labels_asw_=False,
        silhouette_=False,
        hvg_score_=False,
        graph_conn_=False,
        pcr_=False,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=False,
        ari_=False,
        cell_cycle_=False,
        kBET_=False,
        ilisi_=True,
        clisi_=True,
        n_cores=n_cores
    )
    results.rename(index={'cLISI' : "cLISI_celltype"}, inplace=True)
    result_dict = results[0].to_dict()
    result_dict = {k: round(v, 4) for k, v in result_dict.items() if not np.isnan(v)}
    results_disease = scib.metrics.metrics(
    adata,
    adata_int=adata,
    batch_key=batch_key,
    label_key=disease_key,     # Use disease as label
    embed=embed,
    isolated_labels_asw_=False,
    silhouette_=False,
    hvg_score_=False,
    graph_conn_=False,       # Already computed
    pcr_=False,
    isolated_labels_f1_=False,
    trajectory_=False,
    nmi_=False,
    ari_=False,
    cell_cycle_=False,
    kBET_=False,
    ilisi_=False,            # Already computed
    clisi_=True              # ✅ Disease purity
    )
    result_dict['clisi_disease'] = results_disease.loc['cLISI'].values[0]
    return result_dict

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from scGPT models (pretrained or fine-tuned, with/without batch correction)"
    )
    
    # Required arguments
    parser.add_argument("--adata_path", required=True, help="Path to input AnnData (.h5ad)")
    parser.add_argument("--pretrained_model_dir", required=True, help="Directory with pretrained model files")
    parser.add_argument("--finetuned_weights_path", required=True, help="Path to model weights (.pt file)")
    
    # Optional arguments
    parser.add_argument("--compute_eval", action="store_true", help="Computes scBI based metrics to evaluate embedding space quality")
    parser.add_argument("--gene_key", default="feature_name", help="Column in adata.var with gene names")
    parser.add_argument("--batch_key", default="sample", help="Column in adata.var with batch information")
    parser.add_argument("--cell_type", default=None, help="Optional: filter to specific cell type")
    parser.add_argument("--cell_type_key", default='cell_type', help="Optional: celltype key used for evaluationmetric computation")
    parser.add_argument("--disease_key", default='disease', help="Optional: disease key used for evaluationmetric computation")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes (for fine-tuned models)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda, cuda:0, cuda:1, cpu)")
    parser.add_argument("--max_length", type=int, default=3001, help="Maximum sequence length")
    parser.add_argument("--save_format", type=str, default='npy', help="The output can be saved as numpy array or as h5ad")
    
    # Batch correction arguments
    parser.add_argument("--with_batch_correction", action="store_true", 
                       help="Force batch correction mode (auto-detected if not specified)")
    parser.add_argument("--use_batch_labels", action="store_true",
                       help="Pass batch labels during inference (requires 'batch_id' in adata.obs)")
    
    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()
    adata_path = args.adata_path
    pretrained_model_dir = args.pretrained_model_dir
    finetuned_weights_path = args.finetuned_weights_path
    batch_size = args.batch_size
    device = args.device
    save_format = args.save_format
    batch_key = args.batch_key
    cell_type_key = args.cell_type_key
    disease_key = args.disease_key
    max_length = args.max_length
    num_classes = args.num_classes


    print(f"\n{'='*80}")
    print("scGPT EMBEDDING EXTRACTION")
    print(f"{'='*80}\n")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(adata_path)

    # check for MVC layers
    checkpoint = torch.load(finetuned_weights_path)
    mvc_keys = [k for k in checkpoint.keys() if 'mvc_decoder' in k]
    print(f"MVC keys found: {len(mvc_keys)}")
    print(mvc_keys[:5])  # Show first 5

    # Filter to specific cell type if requested
    if args.cell_type:
        cell_type = args.cell_type
        if 'cell_type' in adata.obs and cell_type in adata.obs['cell_type'].unique():
            adata = adata[adata.obs['cell_type'] == cell_type].copy()
            print(f"Filtered to '{cell_type}': {adata.shape}\n")
        else:
            print(f"⚠️  Warning: cell_type '{cell_type}' not found, using all cells\n")

    
    # Extract embeddings
    cell_embeddings = extract_embeddings(adata,
        pretrained_model_dir, 
        finetuned_weights_path, 
        with_batch_correction=args.with_batch_correction,
        batch_col="sample",
        device=device,
        batch_size=batch_size,
        max_len=max_length,
        num_classes=num_classes
    )

    if args.compute_eval:
        adata.obsm["X_scGPT"] = cell_embeddings
        result_dict = get_embedding_quality_metrics(adata, batch_key=batch_key, celltype_key=cell_type_key, disease_key=disease_key, n_cores=5)
        print(f"{'='*80}")
        print("EMBEDDING EVAL METRICS")
        print(f"{'='*80}")
        for key, value in result_dict.items():
            print(f"{key}:\t {value:.2f}")
        # save eval metrics in fintuned model dir
        weights_dir = Path(finetuned_weights_path).parent
        metrics_path = weights_dir / "scbi_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    # Save embeddings
    print(f"{'='*80}")
    print("SAVING EMBEDDINGS")
    print(f"{'='*80}")
    
    output_path = Path(finetuned_weights_path).parent
    #output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_format == "h5ad":
        adata.obsm["X_scGPT"] = cell_embeddings
        output_path = output_path / 'cell_embedding.h5ad'
        adata.write_h5ad(output_path)
        print(f"✓ Saved to AnnData: {output_path}")
        print(f"  Embeddings stored in adata.obsm['X_scGPT']")
    elif save_format == "npy":
        output_path = output_path / 'cell_embedding.npy'
        np.save(output_path, cell_embeddings)
        print(f"✓ Saved as numpy: {output_path}")
    
    print(f"{'='*80}\n")
    print("✓ COMPLETED SUCCESSFULLY!\n")



if __name__ == "__main__":
    main()