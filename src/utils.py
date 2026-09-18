import scanpy as sc
import rapids_singlecell as rsc
from pathlib import Path
import numpy as np

def calculate_umap(adata, space='observation', embedding_key='X_scGPT'):
    '''
    space : str
        Either 'observation' or 'embedding'
    '''
    if space == 'observation':
        print(f"Computing neighbors in observation space")
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        # Rename to specific key
        key_name = 'X_umap_obs'
        adata.obsm[key_name] = adata.obsm['X_umap'].copy()
        adata.obsm.pop('X_umap', None)
    elif space == 'embedding':
        # Check if embedding exists
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
        
        # we don't need to compute pca in embedding space since it is already compressed
        print(f"Computing neighbors in embedding space ({embedding_key})...")
        sc.pp.neighbors(adata, use_rep=embedding_key)
        sc.tl.umap(adata)
        
        # Rename to specific key
        key_name = 'X_umap_emb'
        adata.obsm[key_name] = adata.obsm['X_umap'].copy()
        if 'X_umap' in adata.obsm:
            adata.obsm.pop('X_umap', None)
    return adata, key_name


def calculate_umap_gpu(adata, space='observation', embedding_key='X_scGPT'):
    key_name = 'X_umap_obs' if space == 'observation' else 'X_umap_emb'
    nkey = f'neighbors_{space}'

    rsc.get.anndata_to_GPU(adata)
    if space == 'observation':
        if 'X_pca' not in adata.obsm:
            rsc.pp.pca(adata)
        rsc.pp.neighbors(adata, key_added=nkey)
    elif space == 'embedding':
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
        rsc.pp.neighbors(adata, use_rep=embedding_key, key_added=nkey)
    else:
        raise ValueError(f"Unknown space: {space}")

    rsc.tl.umap(adata, neighbors_key=nkey, key_added=key_name)
    rsc.get.anndata_to_CPU(adata)
    return adata, key_name