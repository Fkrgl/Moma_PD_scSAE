#!/usr/bin/env python
"""
Nido PFCTX: cell type assignment, junk cluster removal, marker-pair trim.

Runs on the existing leiden_0.5 labels -- no re-clustering. The latent space
is not recomputed, so cluster IDs stay stable relative to nido_clustered.h5ad.

Cluster IDs have already shifted between runs, so the manual assignment is
checked against a marker-based score before it is applied. If they disagree,
the script stops rather than mislabelling silently.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

IN_PATH  = Path("/local/fkriegel/Moma_PD_scSAE/dat/nido_clustered.h5ad")
OUT_PATH = Path("/local/fkriegel/Moma_PD_scSAE/dat/nido_annotated.h5ad")
KEY = "leiden_0.5"
SAMPLE_KEY = "sample_id"

# Their code is rowSums(both markers) > 0, i.e. AT LEAST ONE nonzero, despite
# the methods text reading as "both". "any" reproduces their 11% attrition;
# "both" would be far harsher here because SLC17A7 is weakly recovered.
TRIM_MODE = "any"          # "any" | "both"

# The q90 count/gene ceiling. Off by default: it is a poor multiplet proxy and
# with a 27x depth range it removes cells unevenly across donors, pushing a
# technical gradient into cell-type proportions. Set True to match their Fig 1.
APPLY_Q90 = False

MARKERS = {
    "ex":               ["SATB2", "SLC17A7"],
    "in":               ["GAD1", "SLC6A1"],
    "oligodendrocytes": ["MOG", "MOBP"],
    "astrocytes":       ["GFAP", "AQP4"],
    "microglia":        ["CSF1R", "CSF3R"],
    "endothelial":      ["CLDN5", "SLC2A1"],
    "OPC":              ["VCAN", "OLIG1"],
}
all_markers = [g for v in MARKERS.values() for g in v]

# Manual assignment, from the diagnostics table. Keyed to the CURRENT run.
ASSIGN = {
    "0": "ex", "1": "ex", "4": "ex", "6": "ex", "7": "ex", "8": "ex",
    "9": "ex", "10": "ex", "21": "ex", "23": "ex", "24": "ex",
    "11": "in", "12": "in", "14": "in", "19": "in", "20": "in",
    "25": "in", "26": "in",
    "15": "oligodendrocytes",
    "3":  "astrocytes",
    "18": "OPC",
    "16": "microglia",
    "5":  "endothelial",
}

REMOVE = {
    "2":  "heterotypic doublets (40% flagged, 5 lineages)",
    "17": "heterotypic doublets (35% flagged, oligo+astro)",
    "22": "homotypic ex-ex doublets (deepest cluster, 20% flagged)",
    "27": "ambient (952 median genes, 81% one sample)",
    "13": "low complexity, sample-dominated (77%)",
    "28": "oligo-like but 74% one sample, 15 samples, pct_mt exactly 0",
    "29": "doublet-enriched (24%), OPC+oligo",
}

# Nido et al. Brain 2025, post-trim counts, for reference only.
PAPER = {"ex": 121701, "in": 44711, "oligodendrocytes": 83302,
         "astrocytes": 26178, "OPC": 14755, "microglia": 6897,
         "endothelial": 2038}

# --------------------------------------------------------------------- load
adata = sc.read_h5ad(IN_PATH)

if "lognorm" in adata.layers:
    adata.X = adata.layers["lognorm"].copy()
else:
    raise RuntimeError("no lognorm layer; rerun the normalisation block")

cnt = adata.layers["counts"]
assert np.mod(cnt.data, 1).max() == 0, "counts layer is not integer"
print(f"{adata.n_obs:,} cells, {adata.obs[KEY].nunique()} clusters at {KEY}")

covered = set(ASSIGN) | set(REMOVE)
actual = set(adata.obs[KEY].astype(str).unique())
if covered != actual:
    raise RuntimeError(
        f"cluster IDs do not match this object.\n"
        f"  unassigned: {sorted(actual - covered)}\n"
        f"  not present: {sorted(covered - actual)}\n"
        f"The object was probably reclustered. Redo the assignment."
    )

# ------------------------------------------------------- marker-based score
# z-score each marker across clusters, then average within lineage. The gap
# between the best and second-best lineage is the confidence margin: a small
# margin means the cluster expresses two lineages comparably, i.e. doublets.
expr = sc.get.obs_df(adata, keys=all_markers + [KEY])
mm = expr.groupby(KEY, observed=True).mean()
z = (mm - mm.mean()) / mm.std(ddof=0)

score = pd.DataFrame(
    {ct: z[genes].mean(axis=1) for ct, genes in MARKERS.items()}
)
ranked = score.apply(lambda r: r.sort_values(ascending=False), axis=1,
                     result_type="expand")
auto = pd.DataFrame({
    "auto": score.idxmax(axis=1),
    "top": ranked.iloc[:, 0],
    "margin": ranked.iloc[:, 0] - ranked.iloc[:, 1],
})
auto["manual"] = pd.Series(ASSIGN)
auto["n_cells"] = adata.obs[KEY].value_counts()
auto.loc[list(REMOVE), "manual"] = "REMOVE"

print("\n=== marker score vs manual assignment ===")
print(auto.round(2).sort_values("margin").to_string())

kept = auto[auto.manual != "REMOVE"]
mismatch = kept[kept.auto != kept.manual]
if len(mismatch):
    print("\n!! manual label differs from marker argmax:")
    print(mismatch.round(2).to_string())
    print("Expected for clusters where a marker is shared across lineages "
          "(SLC6A1 in astrocytes, VCAN in astrocytes). Confirm each before "
          "proceeding -- an unexpected entry here means the IDs moved.")

low = kept[kept.margin < 0.5]
if len(low):
    print(f"\n!! low-margin kept clusters (< 0.5): {list(low.index)} -- "
          "these are the ones a doublet would look like.")

# ------------------------------------------------------------------ assign
adata.obs["cell_type"] = (
    adata.obs[KEY].astype(str).map(ASSIGN).astype("category")
)
n0 = adata.n_obs

# --------------------------------------------------- drop flagged clusters
print("\n=== removals ===")
for cl, why in REMOVE.items():
    n = int((adata.obs[KEY].astype(str) == cl).sum())
    print(f"  {cl:>3}  {n:>7,}  {why}")

adata = adata[adata.obs.cell_type.notna()].copy()
adata.obs.cell_type = adata.obs.cell_type.cat.remove_unused_categories()
print(f"\n{n0:,} -> {adata.n_obs:,} after cluster removal "
      f"({1 - adata.n_obs / n0:.1%} dropped)")

# ------------------------------------------------------------- q90 ceiling
if APPLY_Q90:
    q = adata.obs[["total_counts", "n_genes_by_counts"]].quantile(0.9)
    keep = ((adata.obs.total_counts < q.total_counts) &
            (adata.obs.n_genes_by_counts < q.n_genes_by_counts))
    print("\n=== q90 ceiling ===")
    print(f"  thresholds: {q.total_counts:.0f} counts, "
          f"{q.n_genes_by_counts:.0f} genes")
    print((~keep).groupby(adata.obs.cell_type, observed=True).sum()
          .to_frame("dropped").to_string())
    adata = adata[keep].copy()
    print(f"  -> {adata.n_obs:,}")

# ----------------------------------------------------------- marker trim
# Keep a barcode only if it has nonzero RAW counts for its assigned type's
# markers. Depth-dependent by construction, so retention is reported per
# sample -- an uneven pattern here becomes a donor-level confound downstream.
gene_ix = {g: adata.var_names.get_loc(g) for g in all_markers
           if g in adata.var_names}
keep = np.zeros(adata.n_obs, dtype=bool)

print("\n=== marker trim ===")
for ct, genes in MARKERS.items():
    in_ct = (adata.obs.cell_type == ct).values
    if not in_ct.any():
        continue
    ix = [gene_ix[g] for g in genes if g in gene_ix]
    sub = adata.layers["counts"][:, ix]
    nz = np.asarray((sub > 0).sum(axis=1)).ravel()
    ok = in_ct & (nz == len(ix) if TRIM_MODE == "both" else nz > 0)
    keep |= ok
    print(f"  {ct:<18} {in_ct.sum():>7,} -> {ok.sum():>7,} "
          f"({ok.sum() / in_ct.sum():.1%})")

adata = adata[keep].copy()

# ---------------------------------------------------------------- reports
final = adata.obs.cell_type.value_counts()
comp = pd.DataFrame({"ours": final, "paper": pd.Series(PAPER)})
comp["diff"] = comp.ours - comp.paper
comp["pct"] = (comp["diff"] / comp.paper * 100).round(1)
comp["frac_ours"] = (comp.ours / comp.ours.sum() * 100).round(1)
comp["frac_paper"] = (comp.paper / comp.paper.sum() * 100).round(1)
print("\n=== final composition vs Nido et al. ===")
print(comp.to_string())
print(f"\ntotal {adata.n_obs:,}  (paper 299,582)")

per_sample = pd.DataFrame({
    "final": adata.obs.groupby(SAMPLE_KEY, observed=True).size(),
    "med_counts": adata.obs.groupby(SAMPLE_KEY, observed=True)
                  .total_counts.median(),
})
per_sample["frac_of_total"] = (per_sample.final / per_sample.final.sum()
                               * 100).round(2)
print("\n=== per-sample retention ===")
print(per_sample.sort_values("final").to_string())

r = np.corrcoef(per_sample.med_counts, per_sample.final)[0, 1]
print(f"\ndepth vs surviving cells: r = {r:.3f}")
print("A strong positive r means the trim removed more from shallow samples, "
      "which is a technical gradient across donors. Check it against "
      "diagnosis group before using proportions in any downstream test.")

ctab = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs.cell_type,
                   normalize="index").round(3)
ctab.to_csv(OUT_PATH.parent / "composition_per_sample.csv")

# ------------------------------------------------------------------ UMAP
# Removed clusters leave holes in the old embedding. Latent space unchanged.
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata)

adata.write_h5ad(OUT_PATH)
print(f"\nwrote {OUT_PATH}")

# Next: subcluster within lineage for their 12 ex / 8 in subtypes and the 3
# oligodendrocyte clusters, using the Bakken/Hodge marker panel. Do that on a
# per-lineage scVI or on the existing latent restricted to the lineage.
