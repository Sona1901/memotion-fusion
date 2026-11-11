# Multimodal Meme Sentiment Classification (Memotion 7k)

Predicts whether an internet meme is **negative, neutral or positive** from its **image** and its **overlaid text**, and compares unimodal baselines with three ways of fusing the two modalities.

> ⚠️ **Content warning:** the dataset contains real internet memes, some of which are offensive or crude. The failure-case figures show real examples.

## Approaches compared

| Model | Input | Fusion |
|---|---|---|
| Majority class | none | baseline |
| Image-only | ResNet-50 pooled features (2048-d) | unimodal |
| Text-only | BERT mean-pooled features (768-d) | unimodal |
| Early fusion | LayerNorm(image) ‖ LayerNorm(text) → MLP | feature-level |
| Late fusion (mean / weighted) | average of image-only and text-only probabilities; weight tuned on validation | decision-level |
| Cross-attention | text tokens ↔ 7×7 image regions, bidirectional attention → MLP | intermediate |
| CLIP linear probe | frozen CLIP ViT-B/32 image + text embeddings → logistic regression | **reference only** |

Both backbones are frozen; only the small heads are trained. Each trainable model is run with 3 seeds (42, 43, 44) and reported as mean ± std.

## Results (test set, 3 seeds, mean ± std)

| Model | Test macro-F1 | Accuracy | Balanced acc. |
|---|---|---|---|
| Majority class | 0.248 | 0.594 | 0.333 |
| Image-only | 0.316 ± 0.014 | 0.438 ± 0.007 | 0.318 ± 0.013 |
| Text-only | 0.313 ± 0.002 | 0.442 ± 0.077 | 0.328 ± 0.001 |
| **Early fusion** | **0.334 ± 0.008** | 0.441 ± 0.033 | 0.336 ± 0.005 |
| Late fusion (mean) | 0.320 ± 0.013 | 0.451 ± 0.018 | 0.321 ± 0.014 |
| Late fusion (weighted) | 0.316 ± 0.014 | 0.438 ± 0.007 | 0.318 ± 0.013 |
| Cross-attention | 0.324 ± 0.019 | 0.443 ± 0.052 | 0.327 ± 0.018 |
| CLIP linear probe (reference) | 0.332 | 0.389 | 0.359 |

* Early fusion is the best of our models, but paired bootstrap tests show no gap between fusion and unimodal models is statistically significant (p = 0.14–0.32).
* All trained heads overfit strongly (e.g. early fusion: train macro-F1 0.82 vs validation 0.35); early stopping on validation macro-F1 limits the damage.
* The weighted late-fusion weight tuned on validation went to 1.0 (image only), so it collapses to the image-only model.
* Negative memes (9% of data) are rarely detected (negative-class F1 0.07–0.11).
* Data quality: 1 unreadable image and 25 exact duplicate images removed; 6,966 memes kept (train 4,876 / val 1,045 / test 1,045).
* Runtime on a Colab T4: about 7 minutes end to end.

Full tables are in `results/tables/` and figures in `results/figures/`. `memotion_fusion.ipynb` is saved with all outputs from this run.



## Pipeline

1. **Labels:** the 5-point `overall_sentiment` is merged into 3 classes.
2. **Text:** hand-corrected text is used first, then raw OCR as a fallback, then a `[NO TEXT]` placeholder. URLs and meme-generator watermarks are removed.
3. **Data quality:** every image is decoded once; unreadable files and exact duplicate images (MD5 hash) are removed to avoid leakage between splits.
4. **Split:** stratified 70 / 15 / 15 (seed 42). Validation is used for early stopping, the late-fusion weight and CLIP's `C`; the test set is used only for the final numbers.
5. **Features:** ResNet-50 (ImageNet weights) on the whole meme resized to 224×224 (no centre crop, so text at the edges is kept); BERT-base-uncased with `max_len` chosen from the token-length distribution.
6. **Training:** AdamW, class-weighted cross-entropy, dropout 0.3, early stopping on validation macro-F1 (patience 6).
7. **Evaluation:** macro-F1 (primary), accuracy, balanced accuracy, macro precision / recall, weighted F1, macro ROC-AUC, per-class F1, confusion matrices, bootstrap 95% CIs and paired bootstrap tests, train-vs-validation gap, efficiency, subgroup analysis, failure cases and attention maps.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `tables/data_quality.csv` | data quality checks |
| `tables/table_main_results.csv` | main comparison, mean ± std |
| `tables/table_bootstrap_ci.csv`, `table_significance.csv` | confidence intervals and paired tests |
| `tables/table_overfitting.csv` | train vs validation macro-F1 at the kept epoch |
| `tables/table_efficiency.csv` | parameters, training and inference time |
| `tables/table_subgroups.csv`, `table_prediction_distribution.csv`, `table_complementarity.csv` | bias and error analysis |
| `figures/fig01` – `fig11` | all report figures |
| `config.json`, `key_numbers.json` | settings and headline numbers |

## Repository structure

```
memotion_fusion.ipynb   main notebook (run top to bottom)
build_notebook.py       script that generates the notebook
requirements.txt        dependencies
results/                tables and figures from the final run
README.md               this file
LICENSE                 MIT licence for the code (not the data)
```

## Data and licence

Memotion 7k: https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k, released for SemEval-2020 Task 8. Check the licence shown on the Kaggle page before reusing the data. The data is used here for academic research only; no attempt is made to identify any person shown in a meme.

## References

- Sharma, C. et al. (2020). SemEval-2020 Task 8: Memotion Analysis – the Visuo-Lingual Metaphor! *Proc. SemEval-2020.*
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR.*
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL.*
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS.*
- Lu, J. et al. (2019). ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. *NeurIPS.*
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML.*
- Baltrušaitis, T., Ahuja, C. & Morency, L.-P. (2019). Multimodal Machine Learning: A Survey and Taxonomy. *IEEE TPAMI.*
