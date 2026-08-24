# Availability-aware multimodal learning for MCI prognosis

This repository contains the implementation used for an MSc Artificial Intelligence project on 36-month progression from mild cognitive impairment (MCI) to Alzheimer's disease using multimodal ADNI data.

The final model combines two complementary pathways:

- a cross-modal interaction pathway, which models dependencies between available modalities;
- a modality-specific evidence pathway, which produces uncertainty-aware predictions from individual modalities and combines their evidence.

The implementation explicitly retains participants with incomplete multimodal records. Missing modalities are represented using availability masks rather than imputed measurements. The main experiments compare fixed equal fusion with learned participant-specific fusion and evaluate predictive uncertainty, modality conflict and temporal sensitivity.

## 1. Repository structure

```text
01_data_preparation/
02_model_training/
03_evaluation_and_uncertainty/
04_temporal_sensitivity/
05_utilities/
```

### 1.1 Data preparation

The data-preparation notebooks construct the clinical cohort, preprocess the non-imaging modalities, prepare the MRI branch, align observations to baseline, create availability and feature masks, and produce the fold-specific model inputs.

The recommended execution order is:

```text
00_Non_Imaging_Inventory_and_Coverage.ipynb
01_Demographics_Preprocessing.ipynb
02_MMSE_Preprocessing.ipynb
03_ADAS_Preprocessing.ipynb
04_FAQ_Preprocessing.ipynb
05_CSF_Biomarker_Preprocessing.ipynb
06_Plasma_Biomarker_Preprocessing.ipynb
07_APOE_Preprocessing.ipynb
08_Clinical_Cohort_and_Baseline_Alignment.ipynb
09_MRI_Cohort_and_Manifest_Preparation.ipynb
10_MRI_DICOM_Extraction_and_Inventory.ipynb
11_MRI_NIfTI_Preprocessing_and_QC.ipynb
12_MRI_MNI_N4_SyN_Preprocessing.ipynb
13_Multimodal_Input_Preparation.ipynb
```

### 1.2 Model training

```text
20_Fixed_Equal_Fusion_Training.ipynb
21_Learned_Participant_Fusion_Seed_Sensitivity.ipynb
22_Fixed_Equal_Fusion_Seed_Sensitivity.ipynb
```

The repeated evaluation uses seeds 17, 42 and 73 with five outer folds for each seed.

### 1.3 Evaluation and uncertainty

```text
30_Fixed_Equal_Fusion_CV_Aggregation.ipynb
31_Learned_vs_Fixed_Fusion_Seed_Comparison.ipynb
32_Learned_Fusion_Seed_Consolidation.ipynb
33_Fixed_Fusion_Uncertainty_Attribution_Repeated_Seeds.ipynb
34_Learned_Fusion_Routing_and_Constant_Baseline.ipynb
```

These notebooks aggregate held-out predictions, compare the fusion strategies, and analyse evidential uncertainty and modality conflict.

### 1.4 Temporal sensitivity

```text
40_Temporal_Sensitivity_Prerequisites_Audit.ipynb
41_Prebaseline_Fixed_Equal_Fusion_Training.ipynb
42_Prebaseline_Fixed_Equal_Fusion_CV_Aggregation.ipynb
43_Original_vs_Prebaseline_Paired_Comparison.ipynb
```

This analysis reconstructs a stricter pre-baseline input definition and compares it with the main fixed-fusion experiment.

## 2. Data availability

This repository does not distribute ADNI data.

The notebooks expect the authorised ADNI data and generated intermediate files to be stored separately, for example in Google Drive. Raw ADNI data, participant-level tables, MRI volumes, model checkpoints and generated experiment outputs should not be committed to this repository.

Users wishing to reproduce the full pipeline must obtain ADNI access independently and reproduce the expected directory structure.

## 3. Execution environment

The notebooks were developed in Google Colab using Python and PyTorch with GPU acceleration.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Some MRI preprocessing stages require additional imaging packages and are computationally expensive. GPU-backed execution is recommended for model training.

## 4. Project root

The notebooks use a Google Drive project directory. The original experiments used:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/adni_mri")
```

If the project directory is stored elsewhere, update `PROJECT_ROOT` in the notebook setup cell before running the pipeline.

Some internal directory names and experiment identifiers are retained for compatibility with the saved experimental outputs. They do not change the terminology used to describe the final method.

## 5. Reproducing the main experiments

The final task-ready fold inputs are created by:

```text
01_data_preparation/13_Multimodal_Input_Preparation.ipynb
```

For the repeated fixed equal-fusion experiment, use:

```text
02_model_training/22_Fixed_Equal_Fusion_Seed_Sensitivity.ipynb
```

For the repeated learned participant-specific fusion experiment, use:

```text
02_model_training/21_Learned_Participant_Fusion_Seed_Sensitivity.ipynb
```

Each configurable training notebook runs one selected seed and one selected outer fold at a time. The full evaluation was repeated for seeds 17, 42 and 73 and for all five outer folds.

The matched repeated-seed comparison is performed in:

```text
03_evaluation_and_uncertainty/31_Learned_vs_Fixed_Fusion_Seed_Comparison.ipynb
```

The repeated-seed uncertainty attribution analysis is performed in:

```text
03_evaluation_and_uncertainty/33_Fixed_Fusion_Uncertainty_Attribution_Repeated_Seeds.ipynb
```

## 6. Demonstration

For a short demonstration of one trained model run, the most convenient notebook is:

```text
02_model_training/22_Fixed_Equal_Fusion_Seed_Sensitivity.ipynb
```

A single seed and fold can be selected in the configuration cell. If the corresponding checkpoint already exists in the configured project directory, the notebook can load the saved model and evaluate the held-out test fold without retraining the complete experiment.

The score from one demonstration fold should not be interpreted as the overall dissertation result. The reported results aggregate held-out predictions across five folds and repeated random seeds.

## 7. Outputs

The notebooks write generated artefacts to the external project directory rather than to this repository. These include:

- fold-specific prepared input tables;
- preprocessing manifests and scalers;
- model checkpoints;
- training histories;
- held-out predictions;
- evaluation metrics;
- uncertainty and conflict tables.

This separation keeps restricted data and large generated files outside the source-code repository.

## 8. Standalone executable

This project is implemented as a reproducible research pipeline in Jupyter notebooks rather than as a standalone application. The notebooks document the execution order, configuration and generated outputs required to reproduce the experiments.
