
## scGPT preprocessing

most of the filtering is already done by `explore_data.ipynb`

```
python preprocess_scgpt_improved.py \
  --adata_path /local/fkriegel/Moma_PD_scSAE/dat/nido_annotated.h5ad \
  --path_out /local/fkriegel/Moma_PD_scSAE/dat/nido_scgpt_preprocessed.h5ad \
  --use_key X \
  --n_bins 51
```

# scGPT finetuning
- mgm + gepc loss

```
conda activate scgpt_env
cd /local/fkriegel/Moma_PD_scSAE/src
```
```
python fine_tune_scGPT.py --config=/local/fkriegel/Moma_PD_scSAE/src/config/config_mgm_gepc.yaml
```

# Extract cell embeddings
Here we figure out a good number of epochs to finetune our model

rift:
```
python extract_embeddings.py \
    --adata_path /share/runs/2026/09-18-fkriegel-scFM/dat/nido_scgpt_preprocessed.h5ad \
    --pretrained_model_dir /share/runs/2026/09-18-fkriegel-scFM/scGPT/pretrained/scGPT_human \
    --finetuned_weights_path /share/runs/2026/09-18-fkriegel-scFM/scGPT/finetuned/scGPT_human/nido/mgm_gepc_weight_1_epoch_1/best_model.pt \
    --save_format npy \
    --batch_size 128 \
    --device "cuda:1" \
    --max_length 2000 \
    --batch_key donor_id \
    --cell_type_key cell_type \
    --disease_key disease \
    --compute_eval
``` 

```
python extract_embeddings.py \
    --adata_path /share/runs/2026/09-18-fkriegel-scFM/dat/nido_scgpt_preprocessed.h5ad \
    --pretrained_model_dir /share/runs/2026/09-18-fkriegel-scFM/scGPT/pretrained/scGPT_human \
    --finetuned_weights_path /share/runs/2026/09-18-fkriegel-scFM/scGPT/pretrained/scGPT_human/best_model.pt \
    --save_format npy \
    --batch_size 128 \
    --device "cuda:1" \
    --max_length 2000 \
    --batch_key donor_id \
    --cell_type_key cell_type \
    --disease_key disease \
    --compute_eval
``` 