# Preprocessing for scGPT.
#
# Every step is OPT-IN: if a flag is not passed, that step is not executed and
# the corresponding Preprocessor argument is set to its "skip" sentinel
# (False for the filters/normalize/log1p/hvg, None for binning).

import argparse
import sys

import numpy as np
import scanpy as sc
from scgpt.preprocess import Preprocessor


def parse_arguments():
    """
    Parse command line arguments for scGPT preprocessing.

    Returns:
        argparse.Namespace: Parsed arguments accessible as attributes
    """
    parser = argparse.ArgumentParser(
        description="Preprocess single-cell data for scGPT. "
                    "Each preprocessing step runs only if its flag is given."
    )

    # ----------------------------------------------------------------- I/O
    parser.add_argument(
        "--adata_path",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--path_out",
        type=str,
        required=True,
        help="Path to save preprocessed AnnData file",
    )
    parser.add_argument(
        "--use_key",
        type=str,
        default=None,
        help="Matrix Preprocessor reads from: 'X' or a key in adata.layers. "
             "Unset -> Preprocessor default (None, i.e. adata.X).",
    )

    # ----------------------------------- pre-Preprocessor steps (opt-in) --
    parser.add_argument(
        "--gene_names",
        type=str,
        default=None,
        help="Column in adata.var to use as var_names. Unset -> var_names left as is.",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=None,
        help="Min number of genes per cell (scanpy filter_cells). Unset -> not applied.",
    )
    parser.add_argument(
        "--filter_cell_by_pct_mt",
        type=float,
        default=None,
        help="Max percent mitochondrial counts per cell. Unset -> not applied.",
    )
    parser.add_argument(
        "--key_pct_mt",
        type=str,
        default="percent.mt",
        help="Column in adata.obs holding percent mitochondrial counts. "
             "Only used with --filter_cell_by_pct_mt.",
    )

    # ------------------------------------- Preprocessor steps (opt-in) ----
    parser.add_argument(
        "--filter_gene_by_counts",
        type=int,
        default=None,
        help="Absolute floor: keep genes detected in >= N cells. "
             "Mutually exclusive with --filter_gene_by_fraction. Unset -> not applied.",
    )
    parser.add_argument(
        "--filter_gene_by_fraction",
        type=float,
        default=None,
        help="Relative floor: keep genes detected in >= this fraction of cells "
             "(converted to an absolute count after cell filtering). "
             "Mutually exclusive with --filter_gene_by_counts. Unset -> not applied.",
    )
    parser.add_argument(
        "--filter_cell_by_counts",
        type=int,
        default=None,
        help="Keep cells with >= N total counts. Unset -> not applied.",
    )
    parser.add_argument(
        "--normalize_total",
        type=float,
        default=None,
        help="Target sum for library-size normalization (e.g. 1e4). Unset -> not applied.",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="Apply log1p. Unset -> not applied.",
    )
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=None,
        help="Number of highly variable genes to subset to. Unset -> no HVG subsetting.",
    )
    parser.add_argument(
        "--hvg_flavor",
        type=str,
        default=None,
        choices=["seurat_v3", "seurat", "cell_ranger"],
        help="HVG flavor. Only used with --n_hvg. "
             "Unset -> 'seurat_v3' if --data_is_raw else 'cell_ranger'.",
    )
    parser.add_argument(
        "--hvg_use_key",
        type=str,
        default=None,
        help="Matrix HVG selection reads from (seurat_v3 expects raw counts). "
             "Only used with --n_hvg.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=None,
        help="Number of bins for value binning. Unset -> no binning.",
    )
    parser.add_argument(
        "--data_is_raw",
        action="store_true",
        help="Data are raw counts. Only affects the default HVG flavor.",
    )

    # ------------------------------------------------ result layer keys ---
    parser.add_argument("--result_normed_key", type=str, default="X_normed")
    parser.add_argument("--result_log1p_key", type=str, default="X_log1p")
    parser.add_argument("--result_binned_key", type=str, default="X_binned")

    args = parser.parse_args()

    if args.filter_gene_by_counts is not None and args.filter_gene_by_fraction is not None:
        parser.error(
            "--filter_gene_by_counts and --filter_gene_by_fraction are mutually "
            "exclusive; pass at most one."
        )

    return args


def main():
    """Main preprocessing workflow."""
    args = parse_arguments()

    # ------------------------------------------------------------- load ---
    print(f"Loading data from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    n_cells_in, n_genes_in = adata.n_obs, adata.n_vars
    print(f"Loaded {n_cells_in} cells x {n_genes_in} genes")

    # -------------------------------------- pre-Preprocessor steps --------
    if args.gene_names is not None:
        if args.gene_names not in adata.var.columns:
            sys.exit(f"ERROR: '{args.gene_names}' not in adata.var.columns")
        print(f"Setting var_names from adata.var['{args.gene_names}']...")
        adata.var_names = adata.var[args.gene_names].astype(str).values
        adata.var_names_make_unique()

    if args.min_genes is not None:
        print(f"Filtering cells with < {args.min_genes} genes...")
        sc.pp.filter_cells(adata, min_genes=args.min_genes)
        print(f"  -> {adata.n_obs} cells x {adata.n_vars} genes")

    if args.filter_cell_by_pct_mt is not None:
        if args.key_pct_mt not in adata.obs.columns:
            sys.exit(f"ERROR: '{args.key_pct_mt}' not in adata.obs.columns")
        print(f"Filtering cells with {args.key_pct_mt} > {args.filter_cell_by_pct_mt}...")
        adata = adata[adata.obs[args.key_pct_mt] <= args.filter_cell_by_pct_mt].copy()
        print(f"  -> {adata.n_obs} cells x {adata.n_vars} genes")

    # ------------------------------- resolve Preprocessor arguments -------
    # Sentinels: False = step skipped for the filters/normalize/log1p/hvg,
    # None = step skipped for binning.
    if args.filter_gene_by_counts is not None:
        filter_gene_by_counts = args.filter_gene_by_counts
    elif args.filter_gene_by_fraction is not None:
        # Resolved against the post-cell-filter cell count.
        filter_gene_by_counts = int(np.ceil(adata.n_obs * args.filter_gene_by_fraction))
        print(
            f"Gene fraction {args.filter_gene_by_fraction} of {adata.n_obs} cells "
            f"-> min {filter_gene_by_counts} cells per gene"
        )
    else:
        filter_gene_by_counts = False

    filter_cell_by_counts = (
        args.filter_cell_by_counts if args.filter_cell_by_counts is not None else False
    )
    normalize_total = (
        args.normalize_total if args.normalize_total is not None else False
    )
    subset_hvg = args.n_hvg if args.n_hvg is not None else False

    if args.hvg_flavor is not None:
        hvg_flavor = args.hvg_flavor
    else:
        hvg_flavor = "seurat_v3" if args.data_is_raw else "cell_ranger"

    # ------------------------------------------------ report the plan -----
    steps = []
    if filter_gene_by_counts is not False:
        steps.append(f"filter_gene_by_counts={filter_gene_by_counts}")
    if filter_cell_by_counts is not False:
        steps.append(f"filter_cell_by_counts={filter_cell_by_counts}")
    if normalize_total is not False:
        steps.append(f"normalize_total={normalize_total} -> '{args.result_normed_key}'")
    if args.log1p:
        steps.append(f"log1p -> '{args.result_log1p_key}'")
    if subset_hvg is not False:
        steps.append(f"subset_hvg={subset_hvg} (flavor={hvg_flavor})")
    if args.n_bins is not None:
        steps.append(f"binning={args.n_bins} -> '{args.result_binned_key}'")

    print("\nPreprocessor steps enabled:")
    if steps:
        for s in steps:
            print(f"  - {s}")
    else:
        print("  (none -- Preprocessor is a no-op)")
    print(f"  reading from: {args.use_key or 'adata.X (default)'}\n")

    # --------------------------------------------------- preprocess -------
    preprocessor = Preprocessor(
        use_key=args.use_key,
        filter_gene_by_counts=filter_gene_by_counts,
        filter_cell_by_counts=filter_cell_by_counts,
        normalize_total=normalize_total,
        result_normed_key=args.result_normed_key,
        log1p=args.log1p,
        result_log1p_key=args.result_log1p_key,
        subset_hvg=subset_hvg,
        hvg_use_key=args.hvg_use_key,
        hvg_flavor=hvg_flavor,
        binning=args.n_bins,
        result_binned_key=args.result_binned_key,
    )

    print("Running preprocessing...")
    preprocessor(adata, batch_key=None)
    print(f"After preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")

    # --------------------------------------------------------- save -------
    print(f"Saving to {args.path_out}...")
    adata.write_h5ad(args.path_out)

    n_cells_dropped = n_cells_in - adata.n_obs
    n_genes_dropped = n_genes_in - adata.n_vars
    print(
        f"Cells: {n_cells_dropped}/{n_cells_in} removed "
        f"({n_cells_dropped / n_cells_in * 100:.2f}%)"
    )
    print(
        f"Genes: {n_genes_dropped}/{n_genes_in} removed "
        f"({n_genes_dropped / n_genes_in * 100:.2f}%)"
    )
    print("Done!")


if __name__ == "__main__":
    main()
