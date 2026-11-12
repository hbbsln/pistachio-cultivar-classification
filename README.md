# Duplicate-Aware and Statistically Validated Evaluation of CNN Feature Fusion and Classical Classifiers for Pistachio Cultivar Classification

Code, data splits and results for the manuscript

> Talan T., Aslan H. *Duplicate-Aware and Statistically Validated Evaluation of CNN Feature Fusion and Classical Classifiers for Pistachio Cultivar Classification.* Submitted to *Journal of Food Measurement and Characterization*.

The study classifies the Siirt and Kirmizi pistachio cultivars with hybrid models that combine features of frozen ImageNet CNNs (ResNet50, EfficientNetB0, MobileNetV2) with classical classifiers (SVM, RF, KNN). It compares RGB and grayscale inputs, RGB–grayscale fusion, validation-based weighted multi-CNN fusion, hand-crafted baselines (HOG, LBP) and end-to-end fine-tuned CNNs.

## Key point: near-duplicate images

The public Pistachio Image Dataset contains **115 pairs (230 images, 10.7 %) of visually identical images** with different file names and hashes. With an image-level split, identical copies fall into both training and test sets. All experiments in this repository therefore use **group-aware splits**: the two copies of a pair are always in the same subset. The list of pairs is in `results_v5_1/splits/near_duplicate_pairs.csv`.

## Repository structure

```
pistachio_classification_torch_v5_1.ipynb      Main analysis (all experiments except the add-on)
pistachio_v5_1_addon_robustness_cost.ipynb     Robustness to acquisition changes and training cost
supplementary/
  pistachio_classification_torch_v5_image_level.ipynb   Image-level split, only for Table 5
figures_manuscript/
  make_schematic_figures.py                    Draws Fig. 1 (workflow) and Fig. 4 (CNN blocks)
results_v5_1/
  splits/        file list, duplicate groups, near-duplicate pairs, split indices (5 seeds), CV fold membership
  tables/        all result tables (CSV) and raw per-run results
  figures/       all result figures (300 dpi)
  predictions/   per-image test predictions of every model and seed; per-image attribution scores
  addon/         robustness and training-cost results
  environment.json, summary_for_manuscript.json
run_logs/        HTML exports of the executed notebooks (outputs of the reported run)
```

Cached CNN features (`results_v5_1/features/`) and fine-tuned checkpoints (`results_v5_1/checkpoints/`) are not included because of their size; the notebooks recreate them.

## Data

Pistachio Image Dataset (Özkan et al., 2021): <https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset>
2,148 RGB images (600 × 600), Kirmizi 1,232, Siirt 916. Download it and set `CONFIG['data_dir']` in the first code cell of the main notebook to the folder that contains the two class folders. Files are read in sorted order, so the split indices in `results_v5_1/splits/split_indices.csv` are reproduced exactly.

## How to run

```bash
pip install -r requirements.txt
```

1. Run `pistachio_classification_torch_v5_1.ipynb` from top to bottom. It writes everything to `results_v5_1/`.
   Approximate runtime on an NVIDIA RTX 3050 Laptop GPU: 2–3 h (about half of it end-to-end fine-tuning; it can be switched off with `CONFIG['run_finetune'] = False`).
2. Run `pistachio_v5_1_addon_robustness_cost.ipynb` (15–20 min). It reuses the features, splits and checkpoints of step 1 and writes to `results_v5_1/addon/`.
3. Optional: `supplementary/pistachio_classification_torch_v5_image_level.ipynb` reproduces the image-level results in Table 5.

Reported run: Python 3.10.19, PyTorch 2.5.1 (CUDA 12.1), torchvision 0.20.1, scikit-learn 1.7.2, Windows, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB). See `results_v5_1/environment.json`.

## Evaluation protocol

* Near-duplicate grouping: cosine similarity ≥ 0.99 of MobileNetV2 features (confirmed with EfficientNetB0 features and pixel differences).
* Group-aware stratified 70/15/15 split, five seeds (42, 0, 123, 7, 2025).
* 5 × 5 repeated stratified group cross-validation on the training + validation set of seed 42; the seed-42 test set is never used for selection or tuning.
* Scaler, PCA (95 % variance), fusion weights and classifiers are fitted on training data only; fusion decisions use validation data only.
* Statistics: corrected resampled t-test (Nadeau & Bengio, 2003) and exact McNemar tests, Holm correction.

## Where to find each table and figure of the manuscript

| Manuscript | File(s) |
|---|---|
| Fig. 1, Fig. 4 | `figures_manuscript/` |
| Fig. 2 | `results_v5_1/figures/fig_samples_masks.png` |
| Fig. 3 | `results_v5_1/figures/fig_near_duplicate_groups.png` |
| Fig. 5 | `results_v5_1/figures/fig_cv_all18_boxplot.png` |
| Fig. 6 | `results_v5_1/figures/fig_pairwise_stats_heatmap.png` |
| Fig. 7 | `results_v5_1/figures/fig_confusion_matrices_all18.png` |
| Fig. 8 | `results_v5_1/figures/fig_roc_pr_best_models.png` |
| Fig. 9 | `results_v5_1/figures/fig_split_consistency.png` |
| Fig. 10 | `results_v5_1/figures/fig_learning_curve.png` |
| Fig. 11 | `results_v5_1/figures/fig_tsne.png` |
| Fig. 12 | `results_v5_1/figures/fig_rgb_gray_fusion.png` |
| Fig. 13 | `results_v5_1/figures/fig_fusion_tau_sensitivity.png` |
| Fig. 14 | `results_v5_1/figures/fig_multicnn_fusion_strategies.png` |
| Fig. 15, Fig. 16 | `results_v5_1/addon/fig_perturbation_examples.png`, `fig_robustness.png` |
| Fig. 17, Fig. 18 | `results_v5_1/figures/fig_xai_examples.png`, `fig_xai_quantitative.png` |
| Fig. 19 | `results_v5_1/figures/fig_shap_pca_components.png` |
| Table 2 | `tables/table_cultivar_morphology.csv` |
| Table 3 | `tables/table_computational_cost.csv`, `tables/table_pca_dimensions.csv` |
| Table 5 | `tables/table_cv_all18_summary.csv`, `tables/table_holdout_test_all18_mean_sd.csv` (group-aware); `run_logs/supplementary_v5_image_level_executed.html` (image-level) |
| Table 6 | `tables/table_cv_all18_summary.csv`, `tables/table_holdout_test_all18_mean_sd.csv` |
| Table 7 | `tables/table_stats_corrected_ttest_all_families.csv` (all 153 pairs and all families) |
| Table 8 | `tables/table_selected_model_seed42_test_with_CI.csv` |
| Table 9 | `tables/table_rgb_gray_fusion_mean_sd.csv`, `tables/table_stats_rgb_gray_fusion_vs_rgb.csv` |
| Table 10 | `tables/table_multicnn_fusion_summary.csv`, `tables/table_multicnn_fusion_weights_and_selection.csv` |
| Table 11 | `tables/table_gridsearch_svm.csv` |
| Table 12 | `tables/table_finetuned_cnns_mean_sd.csv`, `tables/table_handcrafted_baselines_mean_sd.csv`, `tables/table_augmentation_selected_model.csv` |
| Table 13 | `tables/table_mcnemar_selected_vs_handcrafted.csv`, `tables/table_mcnemar_selected_vs_finetuned.csv` |
| Table 14 | `addon/table_robustness_summary.csv` |
| Table 15 | `addon/table_training_cost.csv` |
| Table 16 | `tables/table_xai_quantitative.csv` |
| All metrics of all models | `tables/table_MASTER_all_experiments.csv`, `tables/raw_*.csv` |

## Citation

If you use this code or the duplicate list, please cite the article (citation will be added after publication) and the dataset:

Özkan, I. A., Köklü, M., & Saraçoğlu, R. (2021). Classification of pistachio species using improved k-NN classifier. *Progress in Nutrition*, 23(2), e2021044. https://doi.org/10.23751/pn.v23i2.9686

## License

Code: MIT License (see `LICENSE`). The dataset is not redistributed here; see its Kaggle page for its terms.

## Contact

Tarık Talan (corresponding author) — tarik.talan@gibtu.edu.tr
