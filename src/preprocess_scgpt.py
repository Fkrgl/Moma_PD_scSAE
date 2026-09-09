# This script perfroms the prepocessing steps necessary for single cell data to be useable for scGPT.

import argparse
import scanpy as sc
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer.gene_tokenizer import GeneVocab


def parse_arguments():
    """
    Parse command line arguments for scGPT preprocessing.
    
    Returns:
        argparse.Namespace: Parsed arguments accessible as attributes
    """
    parser = argparse.ArgumentParser(
        description="Preprocess single-cell data for scGPT"
    )
    
    # Required arguments
    parser.add_argument(
        "--adata_path",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)"
    )
    parser.add_argument(
        "--data_is_raw",
        action="store_true",
        help="Whether data is raw (needs normalization + log1p)"
    )
    parser.add_argument(
        "--use_key",
        type=str,
        required=True,
        help="Key in adata.layers to use as input (e.g., 'X' or 'X_log1p')"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        required=True,
        help="Number of bins for expression binning"
    )
    parser.add_argument(
        "--path_out",
        type=str,
        required=True,
        help="Path to save preprocessed AnnData file"
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum number of genes per cell required to pass filtering"
    )
    
    # Optional arguments
    parser.add_argument(
        "--filter_gene_by_fraction",
        type=float,
        default=0.01,
        help="fraction of cells that should at least express this gene"
    )
    parser.add_argument(
        "--filter_cell_by_counts",
        type=int,
        default=0,
        help="minimal number of counts a cell should have"
    )
    parser.add_argument(
        "--filter_cell_by_pct_mt",
        type=int,
        default=None,
        help="maximal percent of mitochanidrial genes a cell should have"
    )
    parser.add_argument(
        "--key_pct_mt",
        type=str,
        default='percent.mt',
        help="key in adata object that hold percentage of mt genes"
    )
    parser.add_argument(
        "--subset_hvg",
        action="store_true",
        help="Whether to subset to highly variable genes"
    )
    parser.add_argument(
        "--do_normalize_total",
        type=float,
        default=0,
        help="Target sum for normalization (default: 1e4, use 0 to skip)"
    )
    parser.add_argument(
        "--do_log1p",
        action="store_true",
        help="Whether to apply log1p transformation"
    )
    parser.add_argument(
        "--gene_names",
        type=str,
        default='feature_name',
        help="column of gene names"
    )
    
    return parser.parse_args()


def main():
    """Main preprocessing workflow."""
    args = parse_arguments()
    
    # Load data
    print(f"Loading data from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    N_cells_unporcessed = adata.n_obs
    # Configure preprocessor
    normalize_total = args.do_normalize_total if args.do_normalize_total > 0 else None
    hvg_flavor = "seurat_v3" if args.data_is_raw else "cell_ranger"
    # select gene names as adata.var index
    if args.gene_names:
        adata.var_names = adata.var[args.gene_names].astype(str).values
        adata.var_names_make_unique()
    if args.min_genes > 0:
        print(f"Filtering cells with < {args.min_genes} genes...")
        print(type(args.min_genes))
        sc.pp.filter_cells(adata, min_genes=args.min_genes)
        print(f"After cell filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    # filter for mt
    if args.filter_cell_by_pct_mt:
        adata = adata[adata.obs[args.key_pct_mt] <= args.filter_cell_by_pct_mt].copy()
    # determine min_gene cutoff
    min_cells_for_gene = int(adata.n_obs * args.filter_gene_by_fraction)
    # preprocess
    preprocessor = Preprocessor(
        use_key=args.use_key,
        filter_gene_by_counts=min_cells_for_gene,
        filter_cell_by_counts=args.filter_cell_by_counts,
        normalize_total=normalize_total,
        result_normed_key="X_normed",
        log1p=args.do_log1p,
        result_log1p_key="X_log1p",
        subset_hvg=args.subset_hvg,
        hvg_flavor=hvg_flavor,
        binning=args.n_bins,
        result_binned_key="X_binned",
    )
    
    print("Running preprocessing...")
    preprocessor(adata, batch_key=None)
    print(f"After preprocessing: {adata.n_obs} cells × {adata.n_vars} genes")
    n_cells_filtered_out = N_cells_unporcessed - adata.n_obs
    # Save preprocessed data
    print(f"Saving to {args.path_out}...")
    adata.write_h5ad(args.path_out)
    print(f"{n_cells_filtered_out} of {adata.n_obs} ({(n_cells_filtered_out/N_cells_unporcessed)*100:.2f}%)were filtered out.")
    print("Done!")


if __name__ == "__main__":
    main()