#!/usr/bin/env python3
"""
JHub worker for one fold/seed of the ADNI 3MT-TMC fixed-equal-fusion
original-capacity experiment.

Built from:
ADNI_3MT_TMC_Fixed_Equal_Fusion_5_Folds_Rebuilt (2).ipynb

The data splits, loss, fixed 0.5/0.5 fusion, modality dropout, optimiser,
scheduler, early stopping, and evaluation logic are preserved. The model
capacity matches the original fixed-equal-fusion architecture.
"""

import argparse
import gc
import json
import math
import os
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--fold", type=int, choices=range(5), required=True)
parser.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
parser.add_argument(
    "--project-root",
    type=str,
    default="/home/jovyan/thesis_data/adni_mri/adni_mri",
)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument(
    "--output-root",
    type=str,
    default="/home/jovyan/thesis_runs",
    help="Writable directory for experiment outputs/checkpoints/results.",
)
args = parser.parse_args()

GLOBAL_RANDOM_SEED = int(args.seed)
SELECTED_FOLD = int(args.fold)
FOLD_RUN_MODE = str(args.mode)
MAXIMUM_EPOCHS = int(args.max_epochs)
BATCH_SIZE = int(args.batch_size)

SELECTED_TASK = "mci_prognosis"
ARCHITECTURE_PRESET = "original_full"
EXPERIMENT_NAME = "gated_cmt_fixed_equal_fusion_full_arch_md050"

PROJECT_ROOT = Path(args.project_root).expanduser().resolve()
OUTPUT_ROOT = Path(args.output_root).expanduser().resolve()
MODEL_ROOT = PROJECT_ROOT / "models" / "3mt_tmc_evidential"
INPUT_ROOT = PROJECT_ROOT / "models" / "final_task_ready_inputs"

SELECTED_INPUT_PATH = (
    INPUT_ROOT
    / SELECTED_TASK
    / f"mci_prognosis_outer_fold_{SELECTED_FOLD}_final_task_ready.csv"
)

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Project root does not exist: {PROJECT_ROOT}")

if not SELECTED_INPUT_PATH.exists():
    raise FileNotFoundError(
        "Prepared fold table was not found:\n"
        f"{SELECTED_INPUT_PATH}"
    )


def set_global_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Seed BEFORE DataLoader creation and model initialisation.
set_global_random_seed(GLOBAL_RANDOM_SEED)

if torch.cuda.is_available():
    # Match the original performance-oriented CUDA policy.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("ADNI 3MT-TMC ORIGINAL-FULL | JHUB")
print("=" * 72)
print(f"Project root: {PROJECT_ROOT}")
print(f"Input table: {SELECTED_INPUT_PATH}")
print(f"Seed: {GLOBAL_RANDOM_SEED}")
print(f"Fold: {SELECTED_FOLD}")
print(f"Mode: {FOLD_RUN_MODE}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


final_model_schema = {
    "identifier_columns": [
        "RID",
        "PTID",
    ],

    "audit_label_columns": [
        "CLINICAL_GROUP",
        "MCI_TRAJECTORY_LABEL",
        "MCI_PROGNOSIS_TARGET",
    ],

    "model_target_column":
        "MODEL_TARGET",

    "split_columns": [
        "OUTER_FOLD",
        "DATA_ROLE",
    ],

    "scaled_continuous_columns": [
        "DEMOGRAPHICS__AGE_AT_BASELINE__Z",
        "DEMOGRAPHICS__PTEDUCAT_CLEAN__Z",
        "ADAS__TOTSCORE__Z",
        "ADAS__TOTAL13__Z",
        "MMSE__MMSE_TOTAL_SCORE__Z",
        "MMSE__MMSE_ORIENTATION_SCORE__Z",
        "MMSE__MMSE_REGISTRATION_SCORE__Z",
        "MMSE__MMSE_ATTENTION_SCORE__Z",
        "MMSE__MMSE_DELAYED_RECALL_SCORE__Z",
        "MMSE__MMSE_LANGUAGE_COMMAND_SCORE__Z",
        "FAQ__FAQTOTAL__Z",
        "CSF__ABETA40__Z",
        "CSF__ABETA42__Z",
        "CSF__TAU__Z",
        "CSF__PTAU__Z",
        "CSF__ABETA42_40_RATIO__Z",
        "PLASMA__pT217_F__Z",
        "PLASMA__AB42_F__Z",
        "PLASMA__AB40_F__Z",
        "PLASMA__AB42_AB40_F__Z",
        "PLASMA__pT217_AB42_F__Z",
        "PLASMA__NfL_Q__Z",
        "PLASMA__GFAP_Q__Z",
        "PLASMA__NfL_F__Z",
        "PLASMA__GFAP_F__Z",
    ],

    "encoded_categorical_columns": [
        "DEMOGRAPHICS__PTGENDER_CLEAN__IDX",
        "DEMOGRAPHICS__PTHAND_CLEAN__IDX",
        "APOE__APOE4_ALLELE_COUNT__IDX",
    ],

    "mri_path_columns": [
        "MRI__NORMALIZED_T1_NPY_PATH",
    ],

    "branch_mask_columns": [
        "BRANCH_MASK__DEMOGRAPHICS",
        "BRANCH_MASK__COGNITIVE_FUNCTIONAL",
        "BRANCH_MASK__CSF",
        "BRANCH_MASK__PLASMA",
        "BRANCH_MASK__APOE",
        "BRANCH_MASK__MRI",
    ],

    "feature_mask_columns": [
        "FEATURE_MASK__DEMOGRAPHICS_AGE_AT_BASELINE",
        "FEATURE_MASK__DEMOGRAPHICS_PTEDUCAT_CLEAN",
        "FEATURE_MASK__DEMOGRAPHICS_PTGENDER_CLEAN",
        "FEATURE_MASK__DEMOGRAPHICS_PTHAND_CLEAN",
        "FEATURE_MASK__ADAS_TOTSCORE",
        "FEATURE_MASK__ADAS_TOTAL13",
        "FEATURE_MASK__MMSE_MMSE_TOTAL_SCORE",
        "FEATURE_MASK__MMSE_MMSE_ORIENTATION_SCORE",
        "FEATURE_MASK__MMSE_MMSE_REGISTRATION_SCORE",
        "FEATURE_MASK__MMSE_MMSE_ATTENTION_SCORE",
        "FEATURE_MASK__MMSE_MMSE_DELAYED_RECALL_SCORE",
        "FEATURE_MASK__MMSE_MMSE_LANGUAGE_COMMAND_SCORE",
        "FEATURE_MASK__FAQ_FAQTOTAL",
        "FEATURE_MASK__CSF_ABETA40",
        "FEATURE_MASK__CSF_ABETA42",
        "FEATURE_MASK__CSF_TAU",
        "FEATURE_MASK__CSF_PTAU",
        "FEATURE_MASK__CSF_ABETA42_40_RATIO",
        "FEATURE_MASK__PLASMA_pT217_F",
        "FEATURE_MASK__PLASMA_AB42_F",
        "FEATURE_MASK__PLASMA_AB40_F",
        "FEATURE_MASK__PLASMA_AB42_AB40_F",
        "FEATURE_MASK__PLASMA_pT217_AB42_F",
        "FEATURE_MASK__PLASMA_NfL_Q",
        "FEATURE_MASK__PLASMA_GFAP_Q",
        "FEATURE_MASK__PLASMA_NfL_F",
        "FEATURE_MASK__PLASMA_GFAP_F",
        "FEATURE_MASK__APOE_APOE4_ALLELE_COUNT",
    ],
}

fold_table = pd.read_csv(SELECTED_INPUT_PATH)

# The transferred CSVs can retain old absolute Google Drive paths.
# Remap only the MRI path column; the prepared arrays themselves are unchanged.
MRI_COLUMN = "MRI__NORMALIZED_T1_NPY_PATH"
MRI_PROCESSED_ROOT = PROJECT_ROOT / "processed"
DEFAULT_MRI_DIR = (
    MRI_PROCESSED_ROOT
    / "mni_n4_syn_rectcrop_177x213x183_t1norm_p01p99"
    / "npy_t1"
)


def remap_transferred_mri_path(value):
    if pd.isna(value):
        return value

    original = Path(str(value))

    if original.exists():
        return str(original)

    parts = original.parts

    if "processed" in parts:
        processed_index = parts.index("processed")
        relative_after_processed = Path(
            *parts[processed_index + 1:]
        )
        candidate = MRI_PROCESSED_ROOT / relative_after_processed
    else:
        candidate = DEFAULT_MRI_DIR / original.name

    return str(candidate)


fold_table[MRI_COLUMN] = (
    fold_table[MRI_COLUMN]
    .apply(remap_transferred_mri_path)
)

available_mri_rows = fold_table["BRANCH_MASK__MRI"].astype(float).eq(1.0)
missing_mri_files = [
    p
    for p in fold_table.loc[available_mri_rows, MRI_COLUMN].dropna().tolist()
    if not Path(str(p)).exists()
]

if missing_mri_files:
    preview = "\n".join(missing_mri_files[:5])
    raise FileNotFoundError(
        f"{len(missing_mri_files)} available-MRI rows point to missing files "
        f"after JHub path remapping. First examples:\n{preview}"
    )

print(
    f"Loaded fold table: {fold_table.shape[0]} rows x "
    f"{fold_table.shape[1]} columns"
)
print(
    f"Available MRI files verified: {int(available_mri_rows.sum())}"
)


# Preparing the model-input groups for fold 0

# Prepared model-input column groups

identifier_columns = final_model_schema[
    "identifier_columns"
]

audit_label_columns = final_model_schema[
    "audit_label_columns"
]

target_column = final_model_schema[
    "model_target_column"
]

split_columns = final_model_schema[
    "split_columns"
]

scaled_continuous_columns = final_model_schema[
    "scaled_continuous_columns"
]

encoded_categorical_columns = final_model_schema[
    "encoded_categorical_columns"
]

mri_path_columns = final_model_schema[
    "mri_path_columns"
]

branch_mask_columns = final_model_schema[
    "branch_mask_columns"
]

feature_mask_columns = final_model_schema[
    "feature_mask_columns"
]


# Fold-0 structure

print("=" * 72)
print(f"FIXED-EQUAL-FUSION FOLD-{SELECTED_FOLD} EXPERIMENT")
print("=" * 72)

print(f"\nExperiment: {EXPERIMENT_NAME}")
print(f"Task: {SELECTED_TASK}")
print(f"Outer fold: {SELECTED_FOLD}")

print("\nPrepared data roles:")
print(
    fold_table["DATA_ROLE"]
    .value_counts()
    .reindex(
        ["train", "validation", "test"]
    )
)

print("\nPrepared target counts by role:")
display(
    pd.crosstab(
        fold_table["DATA_ROLE"],
        fold_table[target_column],
    ).reindex(
        ["train", "validation", "test"]
    )
)

print("\nPrepared input groups:")
print(
    f"- scaled continuous features: "
    f"{len(scaled_continuous_columns)}"
)
print(
    f"- encoded categorical features: "
    f"{len(encoded_categorical_columns)}"
)
print(
    f"- MRI path columns: "
    f"{len(mri_path_columns)}"
)
print(
    f"- branch masks: "
    f"{len(branch_mask_columns)}"
)
print(
    f"- feature masks: "
    f"{len(feature_mask_columns)}"
)

preview_columns = (
    identifier_columns
    + ["CLINICAL_GROUP"]
    + split_columns
    + [target_column]
    + branch_mask_columns
    + mri_path_columns
)

print("\nExample prepared rows:")
display(
    fold_table[
        preview_columns
    ].head(5)
)


# 3. Defining the multimodal dataset-output contract

# Branch-specific predictor columns

# Group the prepared continuous and categorical columns into
# the six modality branches used by the architecture.

dataset_column_contract = {
    "demographics": {
        "continuous": [
            "DEMOGRAPHICS__AGE_AT_BASELINE__Z",
            "DEMOGRAPHICS__PTEDUCAT_CLEAN__Z",
        ],
        "categorical": [
            "DEMOGRAPHICS__PTGENDER_CLEAN__IDX",
            "DEMOGRAPHICS__PTHAND_CLEAN__IDX",
        ],
        "feature_masks": [
            "FEATURE_MASK__DEMOGRAPHICS_AGE_AT_BASELINE",
            "FEATURE_MASK__DEMOGRAPHICS_PTEDUCAT_CLEAN",
            "FEATURE_MASK__DEMOGRAPHICS_PTGENDER_CLEAN",
            "FEATURE_MASK__DEMOGRAPHICS_PTHAND_CLEAN",
        ],
        "branch_mask": "BRANCH_MASK__DEMOGRAPHICS",
    },

    "cognitive_functional": {
        "continuous": [
            "ADAS__TOTSCORE__Z",
            "ADAS__TOTAL13__Z",
            "MMSE__MMSE_TOTAL_SCORE__Z",
            "MMSE__MMSE_ORIENTATION_SCORE__Z",
            "MMSE__MMSE_REGISTRATION_SCORE__Z",
            "MMSE__MMSE_ATTENTION_SCORE__Z",
            "MMSE__MMSE_DELAYED_RECALL_SCORE__Z",
            "MMSE__MMSE_LANGUAGE_COMMAND_SCORE__Z",
            "FAQ__FAQTOTAL__Z",
        ],
        "categorical": [],
        "feature_masks": [
            "FEATURE_MASK__ADAS_TOTSCORE",
            "FEATURE_MASK__ADAS_TOTAL13",
            "FEATURE_MASK__MMSE_MMSE_TOTAL_SCORE",
            "FEATURE_MASK__MMSE_MMSE_ORIENTATION_SCORE",
            "FEATURE_MASK__MMSE_MMSE_REGISTRATION_SCORE",
            "FEATURE_MASK__MMSE_MMSE_ATTENTION_SCORE",
            "FEATURE_MASK__MMSE_MMSE_DELAYED_RECALL_SCORE",
            "FEATURE_MASK__MMSE_MMSE_LANGUAGE_COMMAND_SCORE",
            "FEATURE_MASK__FAQ_FAQTOTAL",
        ],
        "branch_mask": "BRANCH_MASK__COGNITIVE_FUNCTIONAL",
    },

    "csf": {
        "continuous": [
            "CSF__ABETA40__Z",
            "CSF__ABETA42__Z",
            "CSF__TAU__Z",
            "CSF__PTAU__Z",
            "CSF__ABETA42_40_RATIO__Z",
        ],
        "categorical": [],
        "feature_masks": [
            "FEATURE_MASK__CSF_ABETA40",
            "FEATURE_MASK__CSF_ABETA42",
            "FEATURE_MASK__CSF_TAU",
            "FEATURE_MASK__CSF_PTAU",
            "FEATURE_MASK__CSF_ABETA42_40_RATIO",
        ],
        "branch_mask": "BRANCH_MASK__CSF",
    },

    "plasma": {
        "continuous": [
            "PLASMA__pT217_F__Z",
            "PLASMA__AB42_F__Z",
            "PLASMA__AB40_F__Z",
            "PLASMA__AB42_AB40_F__Z",
            "PLASMA__pT217_AB42_F__Z",
            "PLASMA__NfL_Q__Z",
            "PLASMA__GFAP_Q__Z",
            "PLASMA__NfL_F__Z",
            "PLASMA__GFAP_F__Z",
        ],
        "categorical": [],
        "feature_masks": [
            "FEATURE_MASK__PLASMA_pT217_F",
            "FEATURE_MASK__PLASMA_AB42_F",
            "FEATURE_MASK__PLASMA_AB40_F",
            "FEATURE_MASK__PLASMA_AB42_AB40_F",
            "FEATURE_MASK__PLASMA_pT217_AB42_F",
            "FEATURE_MASK__PLASMA_NfL_Q",
            "FEATURE_MASK__PLASMA_GFAP_Q",
            "FEATURE_MASK__PLASMA_NfL_F",
            "FEATURE_MASK__PLASMA_GFAP_F",
        ],
        "branch_mask": "BRANCH_MASK__PLASMA",
    },

    "apoe": {
        "continuous": [],
        "categorical": [
            "APOE__APOE4_ALLELE_COUNT__IDX",
        ],
        "feature_masks": [
            "FEATURE_MASK__APOE_APOE4_ALLELE_COUNT",
        ],
        "branch_mask": "BRANCH_MASK__APOE",
    },

    "mri": {
        "path": "MRI__NORMALIZED_T1_NPY_PATH",
        "branch_mask": "BRANCH_MASK__MRI",
    },
}


# Dataset sample structure

# One participant will later be returned by the PyTorch dataset
# using the following nested structure.
#
# Continuous features will become float32 tensors.
# Categorical indices will become int64 tensors for embeddings.
# Masks will become float32 tensors containing 0 or 1.
# The target will become an int64 class index.

dataset_output_contract = {
    "rid": "Participant RID as an integer",
    "ptid": "Participant PTID as a string",
    "target": "Binary class index: 0 for sMCI and 1 for pMCI",

    "modalities": {
        "demographics": {
            "continuous": "Shape (2,), float32",
            "categorical": "Shape (2,), int64",
            "feature_mask": "Shape (4,), float32",
            "branch_mask": "Scalar, float32",
        },

        "cognitive_functional": {
            "continuous": "Shape (9,), float32",
            "categorical": None,
            "feature_mask": "Shape (9,), float32",
            "branch_mask": "Scalar, float32",
        },

        "csf": {
            "continuous": "Shape (5,), float32",
            "categorical": None,
            "feature_mask": "Shape (5,), float32",
            "branch_mask": "Scalar, float32",
        },

        "plasma": {
            "continuous": "Shape (9,), float32",
            "categorical": None,
            "feature_mask": "Shape (9,), float32",
            "branch_mask": "Scalar, float32",
        },

        "apoe": {
            "continuous": None,
            "categorical": "Shape (1,), int64",
            "feature_mask": "Shape (1,), float32",
            "branch_mask": "Scalar, float32",
        },

        "mri": {
            "path": "Prepared NumPy path or None",
            "image": "Later: shape (1, 177, 213, 183), float32",
            "branch_mask": "Scalar, float32",
        },
    },

    "branch_masks": (
        "Shape (6,), float32, ordered as "
        "demographics, cognitive-functional, CSF, "
        "plasma, APOE, MRI"
    ),

    "feature_masks": (
        "Shape (28,), float32, using the authoritative "
        "feature-mask order"
    ),
}


# Display the agreed contract

print("=" * 72)
print("MULTIMODAL DATASET CONTRACT")
print("=" * 72)

print("\nBranch-specific input dimensions:")

for branch_name, branch_definition in dataset_column_contract.items():

    continuous_count = len(
        branch_definition.get("continuous", [])
    )

    categorical_count = len(
        branch_definition.get("categorical", [])
    )

    feature_mask_count = len(
        branch_definition.get("feature_masks", [])
    )

    has_mri_path = "path" in branch_definition

    print(
        f"- {branch_name}: "
        f"{continuous_count} continuous, "
        f"{categorical_count} categorical, "
        f"{feature_mask_count} feature masks"
        + (", 1 MRI path" if has_mri_path else "")
    )


print("\nPlanned sample output:")
print(
    json.dumps(
        dataset_output_contract,
        indent=2,
    )
)

print(
    "\nThis contract will be used in the next step to "
    "implement the PyTorch dataset."
)

# 4. Implementing the multimodal PyTorch dataset

import numpy as np
import torch

from torch.utils.data import Dataset


# Expected prepared MRI shape

# Preserve the spatial dimensions produced by the completed
# MRI preprocessing pipeline.
MRI_SPATIAL_SHAPE = (177, 213, 183)

# Add one channel dimension when returning an MRI tensor.
MRI_TENSOR_SHAPE = (1, *MRI_SPATIAL_SHAPE)


# Multimodal PyTorch dataset

class ADNIMultimodalDataset(Dataset):
    """
    PyTorch dataset for the prepared ADNI multimodal tables.

    Each participant is returned as a dictionary containing:
    - identifiers;
    - target;
    - separate modality inputs;
    - branch-level masks;
    - feature-level masks.

    MRI arrays are loaded lazily from the prepared NumPy paths.
    """

    def __init__(
        self,
        dataframe,
        column_contract,
        branch_mask_order,
        feature_mask_order,
        target_column,
        load_mri=True,
    ):
        # Reset the row index so that PyTorch sample indices map
        # directly to positional rows in this dataset.
        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        # Retain the prepared column organisation rather than
        # deriving new feature groups from column-name patterns.
        self.column_contract = column_contract

        # Preserve the mask order from the final
        # model-input schema.
        self.branch_mask_order = list(branch_mask_order)
        self.feature_mask_order = list(feature_mask_order)

        self.target_column = target_column

        # This option allows scalar-only experiments and dataset
        # inspection without reading the large MRI arrays.
        self.load_mri = load_mri


    def __len__(self):
        return len(self.dataframe)


    @staticmethod
    def _continuous_tensor(row, columns):
        """
        Convert prepared continuous values to a float32 tensor.
        """

        if not columns:
            return None

        values = (
            row[columns]
            .to_numpy(dtype=np.float32)
        )

        return torch.from_numpy(values)


    @staticmethod
    def _categorical_tensor(row, columns):
        """
        Convert prepared categorical indices to an int64 tensor.
        """

        if not columns:
            return None

        values = (
            row[columns]
            .to_numpy(dtype=np.int64)
        )

        return torch.from_numpy(values)


    @staticmethod
    def _mask_tensor(row, columns):
        """
        Convert prepared binary masks to a float32 tensor.
        """

        if not columns:
            return None

        values = (
            row[columns]
            .to_numpy(dtype=np.float32)
        )

        return torch.from_numpy(values)


    def _load_mri_tensor(
        self,
        mri_path,
        mri_branch_mask,
    ):
        """
        Load one prepared MRI array or return a masked placeholder.
        """

        # A participant without MRI remains in the dataset.
        if float(mri_branch_mask) == 0.0:
            return torch.zeros(
                MRI_TENSOR_SHAPE,
                dtype=torch.float32,
            )

        # Scalar-only inspection can skip disk loading while
        # preserving the same output structure.
        if not self.load_mri:
            return torch.zeros(
                MRI_TENSOR_SHAPE,
                dtype=torch.float32,
            )

        # An available MRI branch should have a prepared path.
        if pd.isna(mri_path):
            raise ValueError(
                "MRI branch mask is 1, but the MRI path is missing."
            )

        mri_path = Path(str(mri_path))

        if not mri_path.exists():
            raise FileNotFoundError(
                f"Prepared MRI array was not found: {mri_path}"
            )

        # Load the already normalised NumPy volume without
        # applying any additional preprocessing.
        mri_array = np.load(
            mri_path,
            allow_pickle=False,
        )

        if mri_array.shape != MRI_SPATIAL_SHAPE:
            raise ValueError(
                "Unexpected MRI shape for "
                f"{mri_path}: {mri_array.shape}"
            )

        # Ensure float32 representation and add the channel axis:
        # (177, 213, 183) -> (1, 177, 213, 183).
        mri_array = np.asarray(
            mri_array,
            dtype=np.float32,
        )

        mri_array = np.expand_dims(
            mri_array,
            axis=0,
        )

        return torch.from_numpy(mri_array)


    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        modalities = {}

        # Demographics

        demographics_contract = self.column_contract[
            "demographics"
        ]

        modalities["demographics"] = {
            "continuous": self._continuous_tensor(
                row,
                demographics_contract["continuous"],
            ),

            "categorical": self._categorical_tensor(
                row,
                demographics_contract["categorical"],
            ),

            "feature_mask": self._mask_tensor(
                row,
                demographics_contract["feature_masks"],
            ),

            "branch_mask": torch.tensor(
                row[
                    demographics_contract["branch_mask"]
                ],
                dtype=torch.float32,
            ),
        }


        # Cognitive and functional measures

        cognitive_contract = self.column_contract[
            "cognitive_functional"
        ]

        modalities["cognitive_functional"] = {
            "continuous": self._continuous_tensor(
                row,
                cognitive_contract["continuous"],
            ),

            "feature_mask": self._mask_tensor(
                row,
                cognitive_contract["feature_masks"],
            ),

            "branch_mask": torch.tensor(
                row[
                    cognitive_contract["branch_mask"]
                ],
                dtype=torch.float32,
            ),
        }


        # CSF

        csf_contract = self.column_contract["csf"]

        modalities["csf"] = {
            "continuous": self._continuous_tensor(
                row,
                csf_contract["continuous"],
            ),

            "feature_mask": self._mask_tensor(
                row,
                csf_contract["feature_masks"],
            ),

            "branch_mask": torch.tensor(
                row[
                    csf_contract["branch_mask"]
                ],
                dtype=torch.float32,
            ),
        }


        # Plasma

        plasma_contract = self.column_contract["plasma"]

        modalities["plasma"] = {
            "continuous": self._continuous_tensor(
                row,
                plasma_contract["continuous"],
            ),

            "feature_mask": self._mask_tensor(
                row,
                plasma_contract["feature_masks"],
            ),

            "branch_mask": torch.tensor(
                row[
                    plasma_contract["branch_mask"]
                ],
                dtype=torch.float32,
            ),
        }


        # APOE

        apoe_contract = self.column_contract["apoe"]

        modalities["apoe"] = {
            "categorical": self._categorical_tensor(
                row,
                apoe_contract["categorical"],
            ),

            "feature_mask": self._mask_tensor(
                row,
                apoe_contract["feature_masks"],
            ),

            "branch_mask": torch.tensor(
                row[
                    apoe_contract["branch_mask"]
                ],
                dtype=torch.float32,
            ),
        }


        # MRI

        mri_contract = self.column_contract["mri"]

        mri_branch_mask = row[
            mri_contract["branch_mask"]
        ]

        mri_path = row[
            mri_contract["path"]
        ]

        modalities["mri"] = {
            "image": self._load_mri_tensor(
                mri_path=mri_path,
                mri_branch_mask=mri_branch_mask,
            ),

            "branch_mask": torch.tensor(
                mri_branch_mask,
                dtype=torch.float32,
            ),
        }


        # Complete sample

        sample = {
            "rid": int(row["RID"]),
            "ptid": str(row["PTID"]),

            "target": torch.tensor(
                int(row[self.target_column]),
                dtype=torch.long,
            ),

            "modalities": modalities,

            "branch_masks": self._mask_tensor(
                row,
                self.branch_mask_order,
            ),

            "feature_masks": self._mask_tensor(
                row,
                self.feature_mask_order,
            ),
        }

        return sample


# Create role-specific datasets

# Retain the prepared role assignments exactly as stored in
# the selected outer-fold table.
train_table = (
    fold_table.loc[
        fold_table["DATA_ROLE"] == "train"
    ]
    .reset_index(drop=True)
)

validation_table = (
    fold_table.loc[
        fold_table["DATA_ROLE"] == "validation"
    ]
    .reset_index(drop=True)
)

test_table = (
    fold_table.loc[
        fold_table["DATA_ROLE"] == "test"
    ]
    .reset_index(drop=True)
)


# Disable MRI disk loading so that the run can inspect the
# dataset structure quickly before constructing the DataLoaders.
train_dataset = ADNIMultimodalDataset(
    dataframe=train_table,
    column_contract=dataset_column_contract,
    branch_mask_order=branch_mask_columns,
    feature_mask_order=feature_mask_columns,
    target_column=target_column,
    load_mri=False,
)

validation_dataset = ADNIMultimodalDataset(
    dataframe=validation_table,
    column_contract=dataset_column_contract,
    branch_mask_order=branch_mask_columns,
    feature_mask_order=feature_mask_columns,
    target_column=target_column,
    load_mri=False,
)

test_dataset = ADNIMultimodalDataset(
    dataframe=test_table,
    column_contract=dataset_column_contract,
    branch_mask_order=branch_mask_columns,
    feature_mask_order=feature_mask_columns,
    target_column=target_column,
    load_mri=False,
)


# Concise dataset summary

print("=" * 72)
print("PYTORCH DATASETS")
print("=" * 72)

print(f"\nTraining participants: {len(train_dataset)}")
print(f"Validation participants: {len(validation_dataset)}")
print(f"Test participants: {len(test_dataset)}")

print(
    "\nThe datasets preserve the prepared role assignments "
    "and return separate modality inputs."
)

print(
    "MRI loading is temporarily disabled for structural "
    "inspection and will be enabled for the DataLoaders."
)

# 6. Constructing the multimodal DataLoaders

from torch.utils.data import DataLoader


# Initial DataLoader settings

# Start with a small batch because each participant may contain
# a full MRI volume with shape (1, 177, 213, 183).
INITIAL_BATCH_SIZE = BATCH_SIZE

# Use the main process for data loading. This is the
# most reliable starting configuration when reading NumPy files
# from mounted Google Drive.
NUM_WORKERS = 0

# Pinned memory can speed transfers to a CUDA device.
PIN_MEMORY = torch.cuda.is_available()


# Recreate the datasets with MRI loading enabled

# Available MRI volumes will now be read lazily when requested.
# Missing MRI branches will retain their zero placeholders and
# branch masks of zero.
train_dataset = ADNIMultimodalDataset(
    dataframe=train_table,
    column_contract=dataset_column_contract,
    branch_mask_order=branch_mask_columns,
    feature_mask_order=feature_mask_columns,
    target_column=target_column,
    load_mri=True,
)

validation_dataset = ADNIMultimodalDataset(
    dataframe=validation_table,
    column_contract=dataset_column_contract,
    branch_mask_order=branch_mask_columns,
    feature_mask_order=feature_mask_columns,
    target_column=target_column,
    load_mri=True,
)

test_dataset = ADNIMultimodalDataset(
    dataframe=test_table,
    column_contract=dataset_column_contract,
    branch_mask_order=branch_mask_columns,
    feature_mask_order=feature_mask_columns,
    target_column=target_column,
    load_mri=True,
)


# Dedicated training-shuffle generator

TRAIN_LOADER_GENERATOR = torch.Generator()
TRAIN_LOADER_GENERATOR.manual_seed(
    GLOBAL_RANDOM_SEED + 1000 * SELECTED_FOLD
)


# Create role-specific DataLoaders

# Shuffle only the training subset.
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=INITIAL_BATCH_SIZE,
    shuffle=True,
    generator=TRAIN_LOADER_GENERATOR,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

# Validation order does not need to be shuffled.
validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=INITIAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

# The test subset also retains a deterministic order.
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=INITIAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)


# Retrieve one complete training batch

example_batch = next(iter(train_loader))


# Display the batched tensor structure

print("=" * 72)
print("EXAMPLE MULTIMODAL BATCH")
print("=" * 72)

print(f"\nBatch size: {example_batch['target'].shape[0]}")
print(f"RID values: {example_batch['rid']}")
print(f"PTID values: {example_batch['ptid']}")
print(f"Targets: {example_batch['target']}")

print("\nModality tensors:")

for modality_name, modality_data in example_batch["modalities"].items():

    print(f"\n{modality_name}")

    for input_name, value in modality_data.items():

        if isinstance(value, torch.Tensor):
            print(
                f"  {input_name}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )
        else:
            print(
                f"  {input_name}: "
                f"type={type(value).__name__}"
            )


print("\nCombined masks:")

print(
    "  branch_masks: "
    f"shape={tuple(example_batch['branch_masks'].shape)}, "
    f"dtype={example_batch['branch_masks'].dtype}"
)

print(
    "  feature_masks: "
    f"shape={tuple(example_batch['feature_masks'].shape)}, "
    f"dtype={example_batch['feature_masks'].dtype}"
)


# Show MRI availability in this batch

batch_mri = example_batch[
    "modalities"
]["mri"]["image"]

batch_mri_masks = example_batch[
    "modalities"
]["mri"]["branch_mask"]

print("\nMRI batch:")

print(
    f"  image shape: {tuple(batch_mri.shape)}"
)

print(
    f"  branch masks: {batch_mri_masks}"
)

print(
    f"  approximate raw MRI batch size: "
    f"{batch_mri.numel() * batch_mri.element_size() / (1024 ** 2):.2f} MB"
)


# DataLoader summary

print("\nDataLoader batches:")

print(
    f"  training: {len(train_loader)} batches"
)

print(
    f"  validation: {len(validation_loader)} batches"
)

print(
    f"  test: {len(test_loader)} batches"
)

print(
    "\nThe complete multimodal batch is ready for "
    "modality-specific encoder construction."
)

# 7. Building the modality-specific encoders

import math

import torch.nn as nn
import torch.nn.functional as F


# Shared representation dimension

# Project every modality into one common latent space so that
# all branches can later enter the same cross-modal transformers.
MODALITY_EMBEDDING_DIM = 64


# Reusable scalar encoder

class MaskAwareScalarEncoder(nn.Module):
    """
    Encode continuous scalar features together with their
    prepared feature-observation masks.
    """

    def __init__(
        self,
        value_dim,
        mask_dim,
        output_dim,
        hidden_dim=64,
        dropout=0.20,
    ):
        super().__init__()

        input_dim = value_dim + mask_dim

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )


    def forward(
        self,
        values,
        feature_mask,
    ):
        # Pass both the prepared values and their masks so that
        # missing placeholders are not treated as genuine observations.
        inputs = torch.cat(
            [
                values,
                feature_mask,
            ],
            dim=-1,
        )

        return self.network(inputs)


# Demographics encoder

class DemographicsEncoder(nn.Module):
    """
    Encode continuous and categorical demographic predictors.
    """

    def __init__(
        self,
        output_dim,
        dropout=0.20,
    ):
        super().__init__()

        # Sex:
        # 0 = missing or unseen;
        # 1 and 2 = observed prepared categories.
        self.sex_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=4,
            padding_idx=0,
        )

        # Handedness:
        # 0 = missing or unseen;
        # 1 and 2 = observed prepared categories.
        self.handedness_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=4,
            padding_idx=0,
        )

        # Input components:
        # 2 continuous values;
        # 4-dimensional sex embedding;
        # 4-dimensional handedness embedding;
        # 4 feature masks.
        input_dim = 2 + 4 + 4 + 4

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                48,
            ),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                48,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )


    def forward(
        self,
        continuous,
        categorical,
        feature_mask,
    ):
        sex_index = categorical[:, 0]
        handedness_index = categorical[:, 1]

        sex_representation = self.sex_embedding(
            sex_index
        )

        handedness_representation = (
            self.handedness_embedding(
                handedness_index
            )
        )

        inputs = torch.cat(
            [
                continuous,
                sex_representation,
                handedness_representation,
                feature_mask,
            ],
            dim=-1,
        )

        return self.network(inputs)


# APOE encoder

class APOEEncoder(nn.Module):
    """
    Encode the prepared APOE epsilon-4 allele-count index.
    """

    def __init__(
        self,
        output_dim,
        dropout=0.10,
    ):
        super().__init__()

        # Prepared APOE indices:
        # 0 = missing;
        # 1 = zero epsilon-4 alleles;
        # 2 = one epsilon-4 allele;
        # 3 = two epsilon-4 alleles.
        self.apoe_embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=8,
            padding_idx=0,
        )

        self.network = nn.Sequential(
            nn.Linear(
                8 + 1,
                32,
            ),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                32,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )


    def forward(
        self,
        categorical,
        feature_mask,
    ):
        apoe_index = categorical[:, 0]

        apoe_representation = self.apoe_embedding(
            apoe_index
        )

        inputs = torch.cat(
            [
                apoe_representation,
                feature_mask,
            ],
            dim=-1,
        )

        return self.network(inputs)


# Residual 3D downsampling block

class ResidualDownsampleBlock3D(nn.Module):
    """
    Downsample a three-dimensional feature map and learn a
    residual representation at the new channel width.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        # The original 3MT image encoder applies spatial
        # downsampling before the residual convolutional paths.
        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2,
        )

        self.main_path = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
            nn.GELU(),

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
        )

        # A point-wise convolution aligns the residual path with
        # the new number of channels.
        self.residual_path = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.activation = nn.GELU()


    def forward(self, inputs):
        pooled_inputs = self.pool(
            inputs
        )

        main_features = self.main_path(
            pooled_inputs
        )

        residual_features = self.residual_path(
            pooled_inputs
        )

        return self.activation(
            main_features
            + residual_features
        )


# 3MT-style CNN-transformer MRI encoder

class MRIEncoder3D(nn.Module):
    """
    Encode the prepared full-volume MRI using a 3D CNN followed
    by a patch-wise transformer encoder.
    """

    def __init__(
        self,
        output_dim,
        input_shape=MRI_SPATIAL_SHAPE,
        patch_embedding_dim=256,
        transformer_heads=8,
        transformer_layers=1,
        transformer_feedforward_dim=512,
        dropout=0.20,
    ):
        super().__init__()

        if patch_embedding_dim % transformer_heads != 0:
            raise ValueError(
                "The MRI patch-embedding dimension must be "
                "divisible by the number of attention heads."
            )

        self.input_shape = tuple(
            input_shape
        )

        self.patch_embedding_dim = (
            patch_embedding_dim
        )

        # Initial convolutional stem

        # Use two initial 3D convolutions, following the broad
        # structure shown in the 3MT image encoder.
        #
        # The first convolution uses stride two because the prepared
        # MRI volumes are larger than the original 3MT inputs.
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                16,
                affine=True,
            ),
            nn.GELU(),

            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                16,
                affine=True,
            ),
            nn.GELU(),
        )


        # Four residual downsampling blocks

        self.residual_blocks = nn.Sequential(
            ResidualDownsampleBlock3D(
                in_channels=16,
                out_channels=32,
            ),

            ResidualDownsampleBlock3D(
                in_channels=32,
                out_channels=64,
            ),

            ResidualDownsampleBlock3D(
                in_channels=64,
                out_channels=128,
            ),

            ResidualDownsampleBlock3D(
                in_channels=128,
                out_channels=256,
            ),
        )


        # Determine the resulting patch grid

        # The stride-two stem convolution applies ceiling division
        # by two for these kernel and padding settings.
        stem_shape = tuple(
            math.ceil(dimension / 2)
            for dimension in self.input_shape
        )

        # Each of the four MaxPool3d layers applies floor division
        # by two.
        patch_grid_shape = stem_shape

        for _ in range(4):
            patch_grid_shape = tuple(
                dimension // 2
                for dimension in patch_grid_shape
            )

        if any(
            dimension < 1
            for dimension in patch_grid_shape
        ):
            raise ValueError(
                "The MRI input becomes too small after "
                "convolutional downsampling."
            )

        self.patch_grid_shape = (
            patch_grid_shape
        )

        self.number_of_patches = math.prod(
            patch_grid_shape
        )


        # Learned positional embeddings

        # Each location in the final 3D feature map becomes one
        # transformer patch token.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.number_of_patches,
                patch_embedding_dim,
            )
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )


        # Patch-wise transformer encoder

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=patch_embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(
                patch_embedding_dim
            ),
        )


        # Projection to the shared modality dimension

        self.projection = nn.Sequential(
            nn.Linear(
                patch_embedding_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


    def forward(self, image):
        # Expected image shape:
        # (batch_size, 1, 177, 213, 183)
        feature_map = self.stem(
            image
        )

        feature_map = self.residual_blocks(
            feature_map
        )

        # Expected feature-map organisation:
        # (batch_size, 256, depth, height, width)
        batch_size, channels, depth, height, width = (
            feature_map.shape
        )

        actual_patch_count = (
            depth
            * height
            * width
        )

        if actual_patch_count != self.number_of_patches:
            raise ValueError(
                "Unexpected MRI patch count. "
                f"Expected {self.number_of_patches}, "
                f"but obtained {actual_patch_count}."
            )

        # Flatten the spatial locations into patch tokens:
        #
        # (B, C, D, H, W)
        # -> (B, C, N)
        # -> (B, N, C)
        patch_tokens = (
            feature_map
            .flatten(start_dim=2)
            .transpose(1, 2)
        )

        # Add learned positional information before modelling
        # relationships between the 3D patch representations.
        patch_tokens = (
            patch_tokens
            + self.position_embedding
        )

        transformed_tokens = (
            self.transformer_encoder(
                patch_tokens
            )
        )

        # The paper applies patch-wise average pooling before the
        # final linear projection.
        pooled_representation = (
            transformed_tokens.mean(
                dim=1
            )
        )

        return self.projection(
            pooled_representation
        )


# Complete set of six modality encoders

class ADNIModalityEncoders(nn.Module):
    """
    Produce one common-dimensional representation per modality.
    """

    def __init__(
        self,
        output_dim=MODALITY_EMBEDDING_DIM,
    ):
        super().__init__()

        self.demographics = DemographicsEncoder(
            output_dim=output_dim,
        )

        self.cognitive_functional = (
            MaskAwareScalarEncoder(
                value_dim=9,
                mask_dim=9,
                hidden_dim=96,
                output_dim=output_dim,
            )
        )

        self.csf = MaskAwareScalarEncoder(
            value_dim=5,
            mask_dim=5,
            hidden_dim=64,
            output_dim=output_dim,
        )

        self.plasma = MaskAwareScalarEncoder(
            value_dim=9,
            mask_dim=9,
            hidden_dim=96,
            output_dim=output_dim,
        )

        self.apoe = APOEEncoder(
            output_dim=output_dim,
        )

        self.mri = MRIEncoder3D(
            output_dim=output_dim,
            input_shape=MRI_SPATIAL_SHAPE,
            patch_embedding_dim=256,
            transformer_heads=8,
            transformer_layers=1,
            transformer_feedforward_dim=512,
            dropout=0.20,
        )


    def forward(self, modalities):
        representations = {}

        representations["demographics"] = (
            self.demographics(
                continuous=modalities[
                    "demographics"
                ]["continuous"],

                categorical=modalities[
                    "demographics"
                ]["categorical"],

                feature_mask=modalities[
                    "demographics"
                ]["feature_mask"],
            )
        )

        representations["cognitive_functional"] = (
            self.cognitive_functional(
                values=modalities[
                    "cognitive_functional"
                ]["continuous"],

                feature_mask=modalities[
                    "cognitive_functional"
                ]["feature_mask"],
            )
        )

        representations["csf"] = self.csf(
            values=modalities[
                "csf"
            ]["continuous"],

            feature_mask=modalities[
                "csf"
            ]["feature_mask"],
        )

        representations["plasma"] = self.plasma(
            values=modalities[
                "plasma"
            ]["continuous"],

            feature_mask=modalities[
                "plasma"
            ]["feature_mask"],
        )

        representations["apoe"] = self.apoe(
            categorical=modalities[
                "apoe"
            ]["categorical"],

            feature_mask=modalities[
                "apoe"
            ]["feature_mask"],
        )

        representations["mri"] = self.mri(
            modalities[
                "mri"
            ]["image"]
        )

        return representations


def count_trainable_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


# 8. Applying branch masks to the encoded modalities

# Fixed modality order

# Use one explicit modality order throughout the architecture.
# This order matches the prepared branch-mask columns.
MODALITY_ORDER = [
    "demographics",
    "cognitive_functional",
    "csf",
    "plasma",
    "apoe",
    "mri",
]


# Mask-aware encoder wrapper

class MaskedADNIModalityEncoders(nn.Module):
    """
    Run the six modality encoders and suppress representations
    from unavailable branches.
    """

    def __init__(
        self,
        output_dim=MODALITY_EMBEDDING_DIM,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.encoders = ADNIModalityEncoders(
            output_dim=output_dim,
        )


    @staticmethod
    def _apply_branch_mask(
        representation,
        branch_mask,
    ):
        """
        Multiply each participant's representation by the
        corresponding scalar branch-availability mask.
        """

        # representation:
        #     (batch_size, embedding_dim)
        #
        # branch_mask:
        #     (batch_size,)
        #
        # Add a final dimension so broadcasting is explicit:
        #     (batch_size,) -> (batch_size, 1)
        expanded_mask = branch_mask.unsqueeze(-1)

        return representation * expanded_mask


    def forward(self, modalities):
        # Get the ordinary encoder outputs.
        raw_representations = self.encoders(
            modalities
        )

        masked_representations = {}

        # Mask every unavailable branch using its own
        # prepared branch-level mask.
        for modality_name in MODALITY_ORDER:

            branch_mask = modalities[
                modality_name
            ]["branch_mask"]

            masked_representations[modality_name] = (
                self._apply_branch_mask(
                    representation=raw_representations[
                        modality_name
                    ],
                    branch_mask=branch_mask,
                )
            )

        # Return both versions for later interpretation and
        # debugging. Only the masked representations should enter
        # multimodal interaction and fusion.
        return {
            "raw": raw_representations,
            "masked": masked_representations,
        }


# 9. Building the availability-gated 3MT cascade

# Fixed cascade order

THREE_MT_CASCADE_ORDER = [
    "demographics",
    "apoe",
    "cognitive_functional",
    "csf",
    "plasma",
    "mri",
]


# One Cascaded Modality Transformer

class CascadedModalityTransformer(nn.Module):
    """
    Apply query self-attention and inject one modality through
    cross-attention.
    """

    def __init__(
        self,
        embedding_dim,
        number_of_heads=4,
        dropout=0.10,
    ):
        super().__init__()

        self.self_attention_norm = nn.LayerNorm(
            embedding_dim
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=number_of_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attention_dropout = nn.Dropout(
            dropout
        )

        self.cross_attention_norm = nn.LayerNorm(
            embedding_dim
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=number_of_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention_dropout = nn.Dropout(
            dropout
        )

        self.output_norm = nn.LayerNorm(
            embedding_dim
        )


    def forward(
        self,
        latent_query,
        modality_embedding,
    ):
        normalised_query = self.self_attention_norm(
            latent_query
        )

        self_attention_output, self_attention_weights = (
            self.self_attention(
                query=normalised_query,
                key=normalised_query,
                value=normalised_query,
                need_weights=True,
                average_attn_weights=False,
            )
        )

        self_attended_query = (
            latent_query
            + self.self_attention_dropout(
                self_attention_output
            )
        )

        normalised_self_query = self.cross_attention_norm(
            self_attended_query
        )

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                query=normalised_self_query,
                key=modality_embedding,
                value=modality_embedding,
                need_weights=True,
                average_attn_weights=False,
            )
        )

        updated_query = (
            self_attended_query
            + self.cross_attention_dropout(
                cross_attention_output
            )
        )

        updated_query = self.output_norm(
            updated_query
        )

        return {
            "updated_query": updated_query,
            "self_attention_weights": self_attention_weights,
            "cross_attention_weights": cross_attention_weights,
        }


# Complete six-stage availability-gated cascade

class ThreeMTCascade(nn.Module):
    """
    Refine one learned latent query through the six CMT stages.

    A stage uses its candidate update only when the corresponding
    effective branch mask is one. Otherwise, the previous query is
    preserved exactly.
    """

    def __init__(
        self,
        embedding_dim=MODALITY_EMBEDDING_DIM,
        modality_order=MODALITY_ORDER,
        cascade_order=THREE_MT_CASCADE_ORDER,
        number_of_heads=4,
        dropout=0.10,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.modality_order = list(modality_order)
        self.cascade_order = list(cascade_order)

        self.modality_to_mask_index = {
            modality_name: modality_index
            for modality_index, modality_name in enumerate(
                self.modality_order
            )
        }

        self.learned_latent_query = nn.Parameter(
            torch.empty(
                1,
                1,
                embedding_dim,
            )
        )

        nn.init.normal_(
            self.learned_latent_query,
            mean=0.0,
            std=0.02,
        )

        self.cmt_blocks = nn.ModuleDict(
            {
                modality_name:
                    CascadedModalityTransformer(
                        embedding_dim=embedding_dim,
                        number_of_heads=number_of_heads,
                        dropout=dropout,
                    )

                for modality_name in self.cascade_order
            }
        )


    def forward(
        self,
        masked_representations,
        branch_masks,
    ):
        first_modality = self.cascade_order[0]

        batch_size = masked_representations[
            first_modality
        ].shape[0]

        latent_query = self.learned_latent_query.expand(
            batch_size,
            -1,
            -1,
        )

        stage_queries = {}
        self_attention_weights = {}
        cross_attention_weights = {}

        for modality_name in self.cascade_order:
            previous_query = latent_query

            modality_token = masked_representations[
                modality_name
            ].unsqueeze(1)

            stage_output = self.cmt_blocks[
                modality_name
            ](
                latent_query=previous_query,
                modality_embedding=modality_token,
            )

            candidate_query = stage_output[
                "updated_query"
            ]

            modality_index = self.modality_to_mask_index[
                modality_name
            ]

            availability = branch_masks[
                :,
                modality_index,
            ].view(
                -1,
                1,
                1,
            ).to(
                dtype=previous_query.dtype
            )

            latent_query = (
                availability * candidate_query
                + (1.0 - availability) * previous_query
            )

            stage_queries[modality_name] = latent_query

            self_attention_weights[modality_name] = (
                stage_output[
                    "self_attention_weights"
                ]
            )

            cross_attention_weights[modality_name] = (
                stage_output[
                    "cross_attention_weights"
                ]
            )

        joint_representation = latent_query.squeeze(
            dim=1
        )

        return {
            "joint_representation":
                joint_representation,

            "stage_queries":
                stage_queries,

            "self_attention_weights":
                self_attention_weights,

            "cross_attention_weights":
                cross_attention_weights,
        }


# 10. Producing independent modality-specific evidential opinions

# Binary prognosis class definition

NUMBER_OF_CLASSES = 2

PROGNOSIS_CLASS_ORDER = [
    "sMCI",
    "pMCI",
]


# One modality-specific evidential head

class EvidentialClassificationHead(nn.Module):
    """
    Convert one modality representation into non-negative class
    evidence and the corresponding Dirichlet opinion.
    """

    def __init__(
        self,
        input_dim,
        number_of_classes,
        hidden_dim=32,
        dropout=0.10,
    ):
        super().__init__()

        self.number_of_classes = number_of_classes

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                number_of_classes,
            ),
        )


    def forward(
        self,
        representation,
        branch_mask,
    ):
        # Use Softplus to obtain non-negative evidence while
        # retaining smooth gradients.
        raw_evidence = F.softplus(
            self.network(
                representation
            )
        )

        # An unavailable modality must not contribute evidence.
        effective_evidence = (
            raw_evidence
            * branch_mask.unsqueeze(-1)
        )

        # Evidence plus one defines the Dirichlet parameters.
        alpha = effective_evidence + 1.0

        strength = alpha.sum(
            dim=-1,
            keepdim=True,
        )

        probabilities = (
            alpha
            / strength
        )

        uncertainty = (
            self.number_of_classes
            / strength
        )

        return {
            "raw_evidence": raw_evidence,
            "evidence": effective_evidence,
            "alpha": alpha,
            "strength": strength,
            "probabilities": probabilities,
            "uncertainty": uncertainty,
        }


# Independent evidential heads for all six modalities

class IndependentModalityEvidentialHeads(nn.Module):
    """
    Produce one independent Dirichlet opinion per modality before
    any 3MT cross-modal interaction.
    """

    def __init__(
        self,
        modality_order,
        input_dim,
        number_of_classes,
    ):
        super().__init__()

        self.modality_order = list(
            modality_order
        )

        self.number_of_classes = (
            number_of_classes
        )

        self.heads = nn.ModuleDict(
            {
                modality_name:
                    EvidentialClassificationHead(
                        input_dim=input_dim,
                        number_of_classes=number_of_classes,
                        hidden_dim=32,
                        dropout=0.10,
                    )

                for modality_name in self.modality_order
            }
        )


    def forward(
        self,
        modality_representations,
        modalities,
    ):
        opinions = {}

        for modality_name in self.modality_order:

            branch_mask = modalities[
                modality_name
            ]["branch_mask"]

            opinions[modality_name] = self.heads[
                modality_name
            ](
                representation=modality_representations[
                    modality_name
                ],
                branch_mask=branch_mask,
            )

        return opinions


# 11. Fusing the independent modality opinions with TMC

# Reduced Dempster-Shafer combination rule

class TMCFusion(nn.Module):
    """
    Fuse independent Dirichlet modality opinions using the
    reduced Dempster-Shafer combination rule used by TMC.
    """

    def __init__(
        self,
        number_of_classes,
        modality_order,
        numerical_epsilon=1e-8,
    ):
        super().__init__()

        self.number_of_classes = (
            number_of_classes
        )

        self.modality_order = list(
            modality_order
        )

        self.numerical_epsilon = (
            numerical_epsilon
        )


    def _dirichlet_to_opinion(
        self,
        alpha,
    ):
        """
        Convert Dirichlet parameters into belief masses and
        one uncertainty mass.
        """

        strength = alpha.sum(
            dim=-1,
            keepdim=True,
        )

        evidence = alpha - 1.0

        belief = evidence / strength

        uncertainty = (
            self.number_of_classes
            / strength
        )

        return {
            "evidence": evidence,
            "strength": strength,
            "belief": belief,
            "uncertainty": uncertainty,
        }


    def _combine_two(
        self,
        alpha_a,
        alpha_b,
    ):
        """
        Combine two batches of Dirichlet opinions.

        Both inputs have shape:
        (batch_size, number_of_classes).
        """

        opinion_a = self._dirichlet_to_opinion(
            alpha_a
        )

        opinion_b = self._dirichlet_to_opinion(
            alpha_b
        )

        belief_a = opinion_a["belief"]
        belief_b = opinion_b["belief"]

        uncertainty_a = opinion_a[
            "uncertainty"
        ]

        uncertainty_b = opinion_b[
            "uncertainty"
        ]


        # Conflict mass

        # The outer product contains every pairwise combination
        # between class beliefs from the two opinions.
        belief_outer_product = (
            belief_a.unsqueeze(-1)
            * belief_b.unsqueeze(-2)
        )

        total_belief_product = (
            belief_outer_product.sum(
                dim=(-2, -1)
            )
        )

        same_class_agreement = (
            torch.diagonal(
                belief_outer_product,
                dim1=-2,
                dim2=-1,
            )
            .sum(dim=-1)
        )

        # Conflict contains products assigned to different classes.
        conflict = (
            total_belief_product
            - same_class_agreement
        )

        normalisation = (
            1.0
            - conflict
        ).clamp_min(
            self.numerical_epsilon
        ).unsqueeze(-1)


        # Fused belief and uncertainty masses

        fused_belief = (
            belief_a * belief_b
            + belief_a * uncertainty_b
            + belief_b * uncertainty_a
        ) / normalisation

        fused_uncertainty = (
            uncertainty_a
            * uncertainty_b
        ) / normalisation


        # Recover the fused Dirichlet opinion

        fused_strength = (
            self.number_of_classes
            / fused_uncertainty.clamp_min(
                self.numerical_epsilon
            )
        )

        fused_evidence = (
            fused_belief
            * fused_strength
        )

        fused_alpha = (
            fused_evidence
            + 1.0
        )

        fused_probabilities = (
            fused_alpha
            / fused_alpha.sum(
                dim=-1,
                keepdim=True,
            )
        )

        return {
            "alpha": fused_alpha,
            "evidence": fused_evidence,
            "belief": fused_belief,
            "uncertainty": fused_uncertainty,
            "strength": fused_strength,
            "probabilities": fused_probabilities,
            "conflict": conflict.unsqueeze(-1),
        }


    def forward(
        self,
        modality_opinions,
    ):
        """
        Sequentially combine the modality-specific opinions in
        the fixed modality order.
        """

        first_modality = self.modality_order[0]

        fused_alpha = modality_opinions[
            first_modality
        ]["alpha"]

        fusion_history = {}

        # Retain the starting opinion so that the complete fusion
        # sequence can later be inspected.
        first_opinion = self._dirichlet_to_opinion(
            fused_alpha
        )

        fusion_history[first_modality] = {
            "alpha": fused_alpha,
            "belief": first_opinion["belief"],
            "uncertainty": first_opinion[
                "uncertainty"
            ],
            "conflict": torch.zeros(
                fused_alpha.shape[0],
                1,
                dtype=fused_alpha.dtype,
                device=fused_alpha.device,
            ),
        }

        for modality_name in self.modality_order[1:]:

            next_alpha = modality_opinions[
                modality_name
            ]["alpha"]

            combined = self._combine_two(
                alpha_a=fused_alpha,
                alpha_b=next_alpha,
            )

            fused_alpha = combined["alpha"]

            fusion_history[modality_name] = {
                "alpha": combined["alpha"],
                "belief": combined["belief"],
                "uncertainty": combined[
                    "uncertainty"
                ],
                "conflict": combined["conflict"],
            }


        # Final fused opinion

        final_strength = fused_alpha.sum(
            dim=-1,
            keepdim=True,
        )

        final_evidence = (
            fused_alpha
            - 1.0
        )

        final_belief = (
            final_evidence
            / final_strength
        )

        final_uncertainty = (
            self.number_of_classes
            / final_strength
        )

        final_probabilities = (
            fused_alpha
            / final_strength
        )

        return {
            "alpha": fused_alpha,
            "evidence": final_evidence,
            "belief": final_belief,
            "strength": final_strength,
            "uncertainty": final_uncertainty,
            "probabilities": final_probabilities,
            "fusion_history": fusion_history,
        }


# 12. Producing the interaction-aware 3MT evidential opinion

# Auxiliary classifier for one intermediate 3MT query

class ThreeMTAuxiliaryClassifier(nn.Module):
    """
    Produce ordinary class logits from one intermediate
    cumulative 3MT query.

    These outputs support training only and are not interpreted
    as independent modality opinions.
    """

    def __init__(
        self,
        input_dim,
        number_of_classes,
        hidden_dim=64,
        dropout=0.10,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LeakyReLU(
                negative_slope=0.01,
            ),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                number_of_classes,
            ),
        )


    def forward(self, representation):
        return self.network(
            representation
        )


# Joint 3MT evidential and auxiliary heads

class ThreeMTPredictionHeads(nn.Module):
    """
    Attach auxiliary classifiers to the intermediate CMT outputs
    and one evidential head to the final 3MT representation.
    """

    def __init__(
        self,
        cascade_order,
        embedding_dim,
        number_of_classes,
    ):
        super().__init__()

        self.cascade_order = list(
            cascade_order
        )

        # The final stage produces the joint evidential opinion.
        self.final_stage = self.cascade_order[-1]

        # Every preceding stage receives an auxiliary classifier.
        self.auxiliary_stages = self.cascade_order[:-1]

        self.auxiliary_heads = nn.ModuleDict(
            {
                stage_name:
                    ThreeMTAuxiliaryClassifier(
                        input_dim=embedding_dim,
                        number_of_classes=number_of_classes,
                        hidden_dim=embedding_dim,
                        dropout=0.10,
                    )

                for stage_name in self.auxiliary_stages
            }
        )

        self.joint_evidential_head = (
            EvidentialClassificationHead(
                input_dim=embedding_dim,
                number_of_classes=number_of_classes,
                hidden_dim=32,
                dropout=0.10,
            )
        )


    def forward(
        self,
        three_mt_output,
    ):
        auxiliary_logits = {}

        # Intermediate auxiliary predictions

        for stage_name in self.auxiliary_stages:

            # Each stored query has shape:
            # (batch_size, 1, embedding_dim).
            stage_representation = three_mt_output[
                "stage_queries"
            ][stage_name].squeeze(1)

            auxiliary_logits[stage_name] = (
                self.auxiliary_heads[
                    stage_name
                ](
                    stage_representation
                )
            )


        # Final joint evidential opinion

        joint_representation = three_mt_output[
            "joint_representation"
        ]

        # The final 3MT query always exists, even when some input
        # modalities are unavailable, so use a branch mask
        # of one for the joint interaction-aware opinion.
        joint_presence_mask = torch.ones(
            joint_representation.shape[0],
            dtype=joint_representation.dtype,
            device=joint_representation.device,
        )

        joint_opinion = self.joint_evidential_head(
            representation=joint_representation,
            branch_mask=joint_presence_mask,
        )

        return {
            "auxiliary_logits":
                auxiliary_logits,

            "joint_opinion":
                joint_opinion,
        }


# 13. Combining 3MT and TMC with fixed equal fusion

def inverse_softplus(value):
    """
    Return an unconstrained value whose Softplus transformation
    is approximately equal to the requested positive value.
    """

    value_tensor = torch.as_tensor(
        value,
        dtype=torch.float32,
    )

    return torch.log(
        torch.expm1(
            value_tensor
        )
    )


class FixedEqualHybridFusion(nn.Module):
    """
    Combine calibrated 3MT and TMC evidence with fixed weights.

    The participant-specific reliability gate is removed:

        w_3MT = 0.5
        w_TMC = 0.5

    The two positive pathway evidence scales remain trainable.
    This isolates the contribution of the learned gate without
    changing the remaining fusion architecture.
    """

    def __init__(
        self,
        number_of_classes,
        number_of_modalities,
        initial_three_mt_scale=1.0,
        initial_tmc_scale=1.0,
    ):
        super().__init__()

        self.number_of_classes = number_of_classes
        self.number_of_modalities = number_of_modalities

        self.three_mt_scale_parameter = nn.Parameter(
            inverse_softplus(
                initial_three_mt_scale
            ).clone()
        )

        self.tmc_scale_parameter = nn.Parameter(
            inverse_softplus(
                initial_tmc_scale
            ).clone()
        )


    def _calculate_mean_available_conflict(
        self,
        tmc_output,
        branch_masks,
        modality_order,
    ):
        """
        Calculate the mean TMC conflict across available fusion stages.
        """

        stage_conflicts = []
        stage_masks = []

        for modality_index, modality_name in enumerate(
            modality_order[1:],
            start=1,
        ):
            stage_conflicts.append(
                tmc_output[
                    "fusion_history"
                ][modality_name]["conflict"]
            )

            stage_masks.append(
                branch_masks[
                    :,
                    modality_index,
                ].unsqueeze(-1)
            )

        stacked_conflicts = torch.stack(
            stage_conflicts,
            dim=1,
        )

        stacked_masks = torch.stack(
            stage_masks,
            dim=1,
        )

        conflict_sum = (
            stacked_conflicts
            * stacked_masks
        ).sum(
            dim=1
        )

        available_fusion_count = (
            stacked_masks.sum(
                dim=1
            ).clamp_min(1.0)
        )

        return (
            conflict_sum
            / available_fusion_count
        )


    def forward(
        self,
        joint_opinion,
        tmc_output,
        branch_masks,
        modality_order,
    ):
        three_mt_evidence = joint_opinion[
            "evidence"
        ]

        tmc_evidence = tmc_output[
            "evidence"
        ]

        three_mt_scale = F.softplus(
            self.three_mt_scale_parameter
        )

        tmc_scale = F.softplus(
            self.tmc_scale_parameter
        )

        calibrated_three_mt_evidence = (
            three_mt_scale
            * three_mt_evidence
        )

        calibrated_tmc_evidence = (
            tmc_scale
            * tmc_evidence
        )

        three_mt_uncertainty = joint_opinion[
            "uncertainty"
        ]

        tmc_uncertainty = tmc_output[
            "uncertainty"
        ]

        mean_tmc_conflict = (
            self._calculate_mean_available_conflict(
                tmc_output=tmc_output,
                branch_masks=branch_masks,
                modality_order=modality_order,
            )
        )

        available_modality_count = (
            branch_masks.sum(
                dim=-1,
                keepdim=True,
            )
        )

        available_modality_proportion = (
            available_modality_count
            / float(
                self.number_of_modalities
            )
        )

        three_mt_weight = torch.full_like(
            three_mt_uncertainty,
            fill_value=0.5,
        )

        tmc_weight = torch.full_like(
            tmc_uncertainty,
            fill_value=0.5,
        )

        # These compatibility fields preserve the prediction-table
        # contract used by the learned-gate experiment.
        gate_logit = torch.zeros_like(
            three_mt_weight
        )

        gate_input = torch.cat(
            [
                three_mt_uncertainty.detach(),
                tmc_uncertainty.detach(),
                mean_tmc_conflict.detach(),
                available_modality_proportion,
                branch_masks,
            ],
            dim=-1,
        )

        final_evidence = (
            three_mt_weight
            * calibrated_three_mt_evidence
            +
            tmc_weight
            * calibrated_tmc_evidence
        )

        final_alpha = final_evidence + 1.0

        final_strength = final_alpha.sum(
            dim=-1,
            keepdim=True,
        )

        final_probabilities = (
            final_alpha
            / final_strength
        )

        final_uncertainty = (
            self.number_of_classes
            / final_strength
        )

        return {
            "evidence": final_evidence,
            "alpha": final_alpha,
            "strength": final_strength,
            "probabilities": final_probabilities,
            "uncertainty": final_uncertainty,
            "three_mt_weight": three_mt_weight,
            "tmc_weight": tmc_weight,
            "gate_logit": gate_logit,
            "gate_input": gate_input,
            "mean_tmc_conflict": mean_tmc_conflict,
            "available_modality_count": available_modality_count,
            "three_mt_scale": three_mt_scale,
            "tmc_scale": tmc_scale,
            "calibrated_three_mt_evidence":
                calibrated_three_mt_evidence,
            "calibrated_tmc_evidence":
                calibrated_tmc_evidence,
        }


# 14. Assembling the complete end-to-end 3MT-TMC model

class ADNIEvidential3MTTMCModel(nn.Module):
    """
    Complete missing-aware and uncertainty-aware multimodal model.

    The model combines:

    1. six modality-specific encoders;
    2. training-time modality dropout;
    3. independent modality evidential heads;
    4. TMC/Dempster-Shafer fusion;
    5. the availability-gated 3MT interaction pathway;
    6. intermediate 3MT auxiliary classifiers;
    7. a joint 3MT evidential head;
    8. fixed-equal final evidence fusion.
    """

    def __init__(
        self,
        embedding_dim=MODALITY_EMBEDDING_DIM,
        number_of_classes=NUMBER_OF_CLASSES,
        modality_order=MODALITY_ORDER,
        cascade_order=THREE_MT_CASCADE_ORDER,
        modality_dropout_probability=0.50,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.number_of_classes = number_of_classes

        self.modality_order = list(
            modality_order
        )

        self.cascade_order = list(
            cascade_order
        )

        self.number_of_modalities = len(
            self.modality_order
        )

        self.modality_dropout_probability = (
            modality_dropout_probability
        )


        # Modality-specific encoders

        # This wrapper returns both raw and branch-masked modality
        # representations.
        self.modality_encoders = (
            MaskedADNIModalityEncoders(
                output_dim=embedding_dim,
            )
        )


        # Independent modality evidential pathway

        self.independent_evidential_heads = (
            IndependentModalityEvidentialHeads(
                modality_order=self.modality_order,
                input_dim=embedding_dim,
                number_of_classes=number_of_classes,
            )
        )

        self.tmc_fusion = TMCFusion(
            number_of_classes=number_of_classes,
            modality_order=self.modality_order,
        )


        # Interaction-aware 3MT pathway

        self.three_mt_cascade = ThreeMTCascade(
            embedding_dim=embedding_dim,
            modality_order=self.modality_order,
            cascade_order=self.cascade_order,
            number_of_heads=4,
            dropout=0.10,
        )

        self.three_mt_prediction_heads = (
            ThreeMTPredictionHeads(
                cascade_order=self.cascade_order,
                embedding_dim=embedding_dim,
                number_of_classes=number_of_classes,
            )
        )


        # Final fixed 50/50 hybrid fusion

        self.hybrid_fusion = (
            FixedEqualHybridFusion(
                number_of_classes=number_of_classes,
                number_of_modalities=self.number_of_modalities,
                initial_three_mt_scale=1.0,
                initial_tmc_scale=1.0,
            )
        )


    # Training-time modality dropout

    def _apply_modality_dropout(
        self,
        original_branch_masks,
    ):
        """
        Randomly hide genuinely available modalities during training.

        Parameters
        ----------
        original_branch_masks:
            Tensor of shape:
            (batch_size, number_of_modalities)

        Returns
        -------
        effective_branch_masks:
            Masks after training-time modality dropout.

        dropped_branch_masks:
            Indicators showing which originally available branches
            were hidden by modality dropout.
        """

        # Validation and testing always use the genuine prepared
        # availability pattern.
        if (
            not self.training
            or self.modality_dropout_probability <= 0.0
        ):
            effective_branch_masks = (
                original_branch_masks.clone()
            )

            dropped_branch_masks = torch.zeros_like(
                original_branch_masks
            )

            return (
                effective_branch_masks,
                dropped_branch_masks,
            )


        # Sample branch-retention indicators

        retention_probability = (
            1.0
            - self.modality_dropout_probability
        )

        retention_masks = torch.bernoulli(
            torch.full_like(
                original_branch_masks,
                fill_value=retention_probability,
            )
        )

        # A naturally unavailable modality remains unavailable.
        effective_branch_masks = (
            original_branch_masks
            * retention_masks
        )


        # Prevent complete information removal

        batch_size = original_branch_masks.shape[0]

        for batch_row in range(batch_size):

            originally_available_indices = torch.nonzero(
                original_branch_masks[
                    batch_row
                ] > 0,
                as_tuple=False,
            ).flatten()

            no_effective_modality = (
                effective_branch_masks[
                    batch_row
                ].sum()
                == 0
            )

            if (
                no_effective_modality
                and originally_available_indices.numel() > 0
            ):
                # Restore one branch that was genuinely
                # available for this participant.
                selected_position = torch.randint(
                    low=0,
                    high=originally_available_indices.numel(),
                    size=(1,),
                    device=original_branch_masks.device,
                )

                selected_modality_index = (
                    originally_available_indices[
                        selected_position
                    ].item()
                )

                effective_branch_masks[
                    batch_row,
                    selected_modality_index,
                ] = 1.0


        dropped_branch_masks = (
            original_branch_masks
            - effective_branch_masks
        ).clamp(
            min=0.0,
            max=1.0,
        )

        return (
            effective_branch_masks,
            dropped_branch_masks,
        )


    # Construct effective modality dictionaries

    def _replace_branch_masks(
        self,
        modalities,
        effective_branch_masks,
    ):
        """
        Construct a new modality dictionary containing the
        training-time effective branch masks.
        """

        effective_modalities = {}

        for modality_index, modality_name in enumerate(
            self.modality_order
        ):
            effective_modalities[modality_name] = dict(
                modalities[modality_name]
            )

            effective_modalities[
                modality_name
            ]["branch_mask"] = (
                effective_branch_masks[
                    :,
                    modality_index,
                ]
            )

        return effective_modalities


    # Complete forward pass

    def forward(
        self,
        modalities,
        original_branch_masks,
    ):
        """
        Run the complete multimodal architecture.

        Parameters
        ----------
        modalities:
            Nested modality dictionary produced by the dataset.

        original_branch_masks:
            Genuine prepared modality-availability tensor with shape:
            (batch_size, number_of_modalities).
        """

        # Apply training-time modality dropout

        (
            effective_branch_masks,
            dropped_branch_masks,
        ) = self._apply_modality_dropout(
            original_branch_masks
        )

        effective_modalities = (
            self._replace_branch_masks(
                modalities=modalities,
                effective_branch_masks=effective_branch_masks,
            )
        )


        # Encode all six modalities

        encoded_modalities = self.modality_encoders(
            effective_modalities
        )

        # The encoders have already applied their effective branch
        # masks. Use these representations for both pathways.
        masked_representations = encoded_modalities[
            "masked"
        ]


        # Independent modality opinions and TMC fusion

        modality_opinions = (
            self.independent_evidential_heads(
                modality_representations=masked_representations,
                modalities=effective_modalities,
            )
        )

        tmc_output = self.tmc_fusion(
            modality_opinions=modality_opinions
        )


        # Interaction-aware 3MT pathway

        three_mt_output = self.three_mt_cascade(
            masked_representations=masked_representations,
            branch_masks=effective_branch_masks,
        )

        three_mt_predictions = (
            self.three_mt_prediction_heads(
                three_mt_output=three_mt_output
            )
        )

        joint_opinion = three_mt_predictions[
            "joint_opinion"
        ]


        # Final fixed-equal hybrid opinion

        final_output = self.hybrid_fusion(
            joint_opinion=joint_opinion,
            tmc_output=tmc_output,
            branch_masks=effective_branch_masks,
            modality_order=self.modality_order,
        )


        return {
            # Final main prediction
            "final_output":
                final_output,

            # Independent uncertainty pathway
            "modality_opinions":
                modality_opinions,

            "tmc_output":
                tmc_output,

            # Interaction-aware pathway
            "three_mt_output":
                three_mt_output,

            "three_mt_predictions":
                three_mt_predictions,

            "joint_opinion":
                joint_opinion,

            # Encoder outputs
            "encoded_modalities":
                encoded_modalities,

            # Missingness and training-time dropout information
            "original_branch_masks":
                original_branch_masks,

            "effective_branch_masks":
                effective_branch_masks,

            "dropped_branch_masks":
                dropped_branch_masks,
        }


# 15. Defining the complete joint training objective

# KL divergence between a predicted Dirichlet distribution and
# a uniform Dirichlet distribution

def dirichlet_kl_to_uniform(
    alpha,
):
    """
    Calculate:

        KL(Dir(alpha) || Dir(1))

    for every participant in the batch.

    Parameters
    ----------
    alpha:
        Positive Dirichlet parameters with shape:
        (batch_size, number_of_classes)

    Returns
    -------
    Tensor with shape:
        (batch_size,)
    """

    number_of_classes = alpha.shape[-1]

    uniform_alpha = torch.ones_like(
        alpha
    )

    alpha_strength = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    uniform_strength = uniform_alpha.sum(
        dim=-1,
        keepdim=True,
    )

    log_normalisation_ratio = (
        torch.lgamma(alpha_strength)
        - torch.lgamma(uniform_strength)
        - torch.lgamma(alpha).sum(
            dim=-1,
            keepdim=True,
        )
        + torch.lgamma(uniform_alpha).sum(
            dim=-1,
            keepdim=True,
        )
    )

    digamma_difference = (
        torch.digamma(alpha)
        - torch.digamma(alpha_strength)
    )

    parameter_difference = (
        alpha
        - uniform_alpha
    )

    expectation_term = (
        parameter_difference
        * digamma_difference
    ).sum(
        dim=-1,
        keepdim=True,
    )

    kl_divergence = (
        log_normalisation_ratio
        + expectation_term
    )

    return kl_divergence.squeeze(-1)


# Evidential classification loss

def evidential_classification_loss(
    alpha,
    targets,
    number_of_classes,
    annealing_coefficient,
    class_weights=None,
    reduction="mean",
):
    """
    Calculate the expected cross-entropy under a Dirichlet
    distribution together with annealed KL regularisation.

    Parameters
    ----------
    alpha:
        Dirichlet parameters with shape:
        (batch_size, number_of_classes)

    targets:
        Integer class labels with shape:
        (batch_size,)

    number_of_classes:
        Number of prognosis classes.

    annealing_coefficient:
        Current coefficient applied to the KL term.

    class_weights:
        Optional class-weight tensor with shape:
        (number_of_classes,)

    reduction:
        "none", "mean", or "sum".
    """

    targets = targets.long()

    one_hot_targets = F.one_hot(
        targets,
        num_classes=number_of_classes,
    ).to(
        dtype=alpha.dtype
    )

    strength = alpha.sum(
        dim=-1,
        keepdim=True,
    )


    # Expected cross-entropy under the Dirichlet distribution

    expected_cross_entropy_by_class = (
        torch.digamma(strength)
        - torch.digamma(alpha)
    )

    expected_cross_entropy = (
        one_hot_targets
        * expected_cross_entropy_by_class
    ).sum(
        dim=-1
    )


    # Optional class weighting

    if class_weights is not None:

        sample_weights = class_weights[
            targets
        ].to(
            dtype=alpha.dtype,
            device=alpha.device,
        )

        expected_cross_entropy = (
            expected_cross_entropy
            * sample_weights
        )


    # Remove correct-class evidence from the KL penalty

    adjusted_alpha = (
        one_hot_targets
        +
        (
            1.0
            - one_hot_targets
        )
        * alpha
    )

    kl_regularisation = (
        dirichlet_kl_to_uniform(
            adjusted_alpha
        )
    )

    per_sample_loss = (
        expected_cross_entropy
        +
        annealing_coefficient
        * kl_regularisation
    )


    # Requested reduction

    if reduction == "none":
        reduced_loss = per_sample_loss

    elif reduction == "mean":
        reduced_loss = per_sample_loss.mean()

    elif reduction == "sum":
        reduced_loss = per_sample_loss.sum()

    else:
        raise ValueError(
            "reduction must be 'none', 'mean', or 'sum'."
        )


    return {
        "loss":
            reduced_loss,

        "per_sample_loss":
            per_sample_loss,

        "expected_cross_entropy":
            expected_cross_entropy,

        "kl_regularisation":
            kl_regularisation,
    }


# Complete multi-output objective

class HybridEvidentialTrainingLoss(nn.Module):
    """
    Jointly supervise the final hybrid output, both main pathways,
    the individual available modality opinions, and the auxiliary
    3MT classifiers.
    """

    def __init__(
        self,
        number_of_classes,
        modality_order,
        evidential_annealing_epochs=10,
        gate_regularisation_epochs=0,
        joint_loss_weight=0.50,
        tmc_loss_weight=0.50,
        modality_loss_weight=0.10,
        auxiliary_loss_weight=0.10,
        gate_loss_weight=0.0,
        class_weights=None,
    ):
        super().__init__()

        self.number_of_classes = (
            number_of_classes
        )

        self.modality_order = list(
            modality_order
        )

        self.evidential_annealing_epochs = (
            evidential_annealing_epochs
        )

        self.gate_regularisation_epochs = (
            gate_regularisation_epochs
        )

        self.joint_loss_weight = (
            joint_loss_weight
        )

        self.tmc_loss_weight = (
            tmc_loss_weight
        )

        self.modality_loss_weight = (
            modality_loss_weight
        )

        self.auxiliary_loss_weight = (
            auxiliary_loss_weight
        )

        self.gate_loss_weight = (
            gate_loss_weight
        )


        # Optional training-fold class weights

        if class_weights is None:

            self.register_buffer(
                "class_weights",
                None,
            )

        else:

            class_weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )

            if class_weights.shape != (
                number_of_classes,
            ):
                raise ValueError(
                    "class_weights must contain one value "
                    "for each class."
                )

            self.register_buffer(
                "class_weights",
                class_weights,
            )


    # Annealing schedules

    def _evidential_annealing_coefficient(
        self,
        epoch,
    ):
        """
        Increase the KL coefficient linearly from zero to one.
        """

        if self.evidential_annealing_epochs <= 0:
            return 1.0

        return min(
            1.0,
            float(epoch)
            / float(
                self.evidential_annealing_epochs
            ),
        )


    def _gate_annealing_coefficient(
        self,
        epoch,
    ):
        """
        Reduce the gate-balance penalty to zero after the initial
        optimisation period.
        """

        if self.gate_regularisation_epochs <= 0:
            return 0.0

        return max(
            0.0,
            1.0
            -
            (
                float(epoch - 1)
                /
                float(
                    self.gate_regularisation_epochs
                )
            ),
        )


    # Complete loss calculation

    def forward(
        self,
        model_output,
        targets,
        epoch,
    ):
        targets = targets.long()

        evidential_annealing = (
            self._evidential_annealing_coefficient(
                epoch
            )
        )

        gate_annealing = (
            self._gate_annealing_coefficient(
                epoch
            )
        )


        # Final hybrid evidential loss

        final_loss_components = (
            evidential_classification_loss(
                alpha=model_output[
                    "final_output"
                ]["alpha"],

                targets=targets,

                number_of_classes=
                    self.number_of_classes,

                annealing_coefficient=
                    evidential_annealing,

                class_weights=
                    self.class_weights,

                reduction="mean",
            )
        )

        final_loss = final_loss_components[
            "loss"
        ]


        # Interaction-aware 3MT evidential loss

        joint_loss_components = (
            evidential_classification_loss(
                alpha=model_output[
                    "joint_opinion"
                ]["alpha"],

                targets=targets,

                number_of_classes=
                    self.number_of_classes,

                annealing_coefficient=
                    evidential_annealing,

                class_weights=
                    self.class_weights,

                reduction="mean",
            )
        )

        joint_loss = joint_loss_components[
            "loss"
        ]


        # TMC-fused evidential loss

        tmc_loss_components = (
            evidential_classification_loss(
                alpha=model_output[
                    "tmc_output"
                ]["alpha"],

                targets=targets,

                number_of_classes=
                    self.number_of_classes,

                annealing_coefficient=
                    evidential_annealing,

                class_weights=
                    self.class_weights,

                reduction="mean",
            )
        )

        tmc_loss = tmc_loss_components[
            "loss"
        ]


        # Independent available-modality evidential loss

        effective_branch_masks = model_output[
            "effective_branch_masks"
        ]

        weighted_modality_loss_sum = torch.zeros(
            (),
            dtype=final_loss.dtype,
            device=final_loss.device,
        )

        available_opinion_count = torch.zeros(
            (),
            dtype=final_loss.dtype,
            device=final_loss.device,
        )

        modality_loss_by_name = {}

        for modality_index, modality_name in enumerate(
            self.modality_order
        ):

            modality_alpha = model_output[
                "modality_opinions"
            ][modality_name]["alpha"]

            modality_loss_components = (
                evidential_classification_loss(
                    alpha=modality_alpha,

                    targets=targets,

                    number_of_classes=
                        self.number_of_classes,

                    annealing_coefficient=
                        evidential_annealing,

                    class_weights=
                        self.class_weights,

                    reduction="none",
                )
            )

            per_sample_modality_loss = (
                modality_loss_components[
                    "per_sample_loss"
                ]
            )

            modality_mask = effective_branch_masks[
                :,
                modality_index,
            ].to(
                dtype=per_sample_modality_loss.dtype
            )

            masked_modality_loss_sum = (
                per_sample_modality_loss
                * modality_mask
            ).sum()

            modality_available_count = (
                modality_mask.sum()
            )

            weighted_modality_loss_sum = (
                weighted_modality_loss_sum
                + masked_modality_loss_sum
            )

            available_opinion_count = (
                available_opinion_count
                + modality_available_count
            )

            modality_loss_by_name[
                modality_name
            ] = (
                masked_modality_loss_sum
                /
                modality_available_count.clamp_min(
                    1.0
                )
            )

        modality_loss = (
            weighted_modality_loss_sum
            /
            available_opinion_count.clamp_min(
                1.0
            )
        )


        # Intermediate 3MT auxiliary cross-entropy loss

        auxiliary_logits = model_output[
            "three_mt_predictions"
        ]["auxiliary_logits"]

        auxiliary_loss_by_stage = {}

        auxiliary_losses = []

        for stage_name, stage_logits in (
            auxiliary_logits.items()
        ):

            stage_loss = F.cross_entropy(
                input=stage_logits,
                target=targets,
                weight=self.class_weights,
            )

            auxiliary_loss_by_stage[
                stage_name
            ] = stage_loss

            auxiliary_losses.append(
                stage_loss
            )

        if auxiliary_losses:

            auxiliary_loss = torch.stack(
                auxiliary_losses
            ).mean()

        else:

            auxiliary_loss = torch.zeros(
                (),
                dtype=final_loss.dtype,
                device=final_loss.device,
            )


        # Early gate-balance regularisation

        three_mt_weights = model_output[
            "final_output"
        ]["three_mt_weight"]

        raw_gate_loss = (
            three_mt_weights.mean()
            - 0.5
        ).pow(2)

        annealed_gate_loss = (
            gate_annealing
            * raw_gate_loss
        )


        # Weighted total objective

        total_loss = (
            final_loss
            +
            self.joint_loss_weight
            * joint_loss
            +
            self.tmc_loss_weight
            * tmc_loss
            +
            self.modality_loss_weight
            * modality_loss
            +
            self.auxiliary_loss_weight
            * auxiliary_loss
            +
            self.gate_loss_weight
            * annealed_gate_loss
        )


        return {
            "total_loss":
                total_loss,

            "final_loss":
                final_loss,

            "joint_loss":
                joint_loss,

            "tmc_loss":
                tmc_loss,

            "modality_loss":
                modality_loss,

            "auxiliary_loss":
                auxiliary_loss,

            "raw_gate_loss":
                raw_gate_loss,

            "annealed_gate_loss":
                annealed_gate_loss,

            "evidential_annealing":
                torch.tensor(
                    evidential_annealing,
                    dtype=final_loss.dtype,
                    device=final_loss.device,
                ),

            "gate_annealing":
                torch.tensor(
                    gate_annealing,
                    dtype=final_loss.dtype,
                    device=final_loss.device,
                ),

            "available_opinion_count":
                available_opinion_count,

            "modality_loss_by_name":
                modality_loss_by_name,

            "auxiliary_loss_by_stage":
                auxiliary_loss_by_stage,

            "final_expected_cross_entropy":
                final_loss_components[
                    "expected_cross_entropy"
                ].mean(),

            "final_kl_regularisation":
                final_loss_components[
                    "kl_regularisation"
                ].mean(),

            "joint_expected_cross_entropy":
                joint_loss_components[
                    "expected_cross_entropy"
                ].mean(),

            "joint_kl_regularisation":
                joint_loss_components[
                    "kl_regularisation"
                ].mean(),

            "tmc_expected_cross_entropy":
                tmc_loss_components[
                    "expected_cross_entropy"
                ].mean(),

            "tmc_kl_regularisation":
                tmc_loss_components[
                    "kl_regularisation"
                ].mean(),
        }


# Instantiate the reduced-capacity model AFTER seeding

complete_model = ADNIEvidential3MTTMCModel(
    embedding_dim=MODALITY_EMBEDDING_DIM,
    number_of_classes=NUMBER_OF_CLASSES,
    modality_order=MODALITY_ORDER,
    cascade_order=THREE_MT_CASCADE_ORDER,
    modality_dropout_probability=0.50,
)

training_objective = HybridEvidentialTrainingLoss(
    number_of_classes=NUMBER_OF_CLASSES,
    modality_order=MODALITY_ORDER,
    evidential_annealing_epochs=10,
    gate_regularisation_epochs=0,
    joint_loss_weight=0.50,
    tmc_loss_weight=0.50,
    modality_loss_weight=0.10,
    auxiliary_loss_weight=0.10,
    gate_loss_weight=0.0,
    class_weights=None,
)

complete_model = complete_model.to(DEVICE)
training_objective = training_objective.to(DEVICE)

# Preserve the original optimisation settings.
INITIAL_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAXIMUM_GRADIENT_NORM = 5.0
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 3
SCHEDULER_REDUCTION_FACTOR = 0.5
MINIMUM_LEARNING_RATE = 1e-6
CLASSIFICATION_THRESHOLD = 0.50

optimizer = torch.optim.AdamW(
    params=complete_model.parameters(),
    lr=INITIAL_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    factor=SCHEDULER_REDUCTION_FACTOR,
    patience=SCHEDULER_PATIENCE,
    min_lr=MINIMUM_LEARNING_RATE,
)

# Seed-specific output directories prevent the five repeated seeds
# from overwriting one another.
# Keep the transferred dataset read-only. All newly generated checkpoints,
# histories, predictions and summaries live in a separate writable JHub area.
EXPERIMENT_ROOT = OUTPUT_ROOT / EXPERIMENT_NAME
SEED_ROOT = EXPERIMENT_ROOT / f"seed_{GLOBAL_RANDOM_SEED}"
FOLD_TRAINING_DIR = (
    SEED_ROOT
    / SELECTED_TASK
    / f"fold_{SELECTED_FOLD}"
)
CHECKPOINT_DIR = FOLD_TRAINING_DIR / "checkpoints"
HISTORY_DIR = FOLD_TRAINING_DIR / "history"
PREDICTION_DIR = FOLD_TRAINING_DIR / "predictions"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_validation_auc_checkpoint.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_epoch_checkpoint.pt"
TRAINING_HISTORY_PATH = HISTORY_DIR / "training_history.csv"
TRAINING_CONFIGURATION_PATH = HISTORY_DIR / "training_configuration.json"
VALIDATION_PREDICTIONS_PATH = PREDICTION_DIR / "best_validation_predictions.csv"
TEST_PREDICTIONS_PATH = PREDICTION_DIR / "test_predictions.csv"

existing_fold_files = []
if FOLD_TRAINING_DIR.exists():
    existing_fold_files = [
        path
        for path in FOLD_TRAINING_DIR.rglob("*")
        if path.is_file()
    ]

if FOLD_RUN_MODE == "fresh" and existing_fold_files:
    raise FileExistsError(
        "Fresh training was requested, but this seed/fold directory "
        "already contains files. Use --mode resume for an interrupted "
        "run or remove the intentionally discarded directory.\n"
        f"{FOLD_TRAINING_DIR}"
    )

if FOLD_RUN_MODE == "resume" and not LAST_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        "Resume mode was requested but no last-epoch checkpoint exists:\n"
        f"{LAST_CHECKPOINT_PATH}"
    )

for directory_path in [
    EXPERIMENT_ROOT,
    SEED_ROOT,
    FOLD_TRAINING_DIR,
    CHECKPOINT_DIR,
    HISTORY_DIR,
    PREDICTION_DIR,
]:
    directory_path.mkdir(parents=True, exist_ok=True)

training_configuration = {
    "experiment_name": EXPERIMENT_NAME,
    "project_root": str(PROJECT_ROOT),
    "output_root": str(OUTPUT_ROOT),
    "architecture_preset": ARCHITECTURE_PRESET,
    "availability_gated_cmt": True,
    "fixed_equal_fusion": True,
    "three_mt_weight": 0.5,
    "tmc_weight": 0.5,
    "task": SELECTED_TASK,
    "fold": int(SELECTED_FOLD),
    "random_seed": int(GLOBAL_RANDOM_SEED),
    "maximum_epochs": int(MAXIMUM_EPOCHS),
    "batch_size": int(train_loader.batch_size),
    "initial_learning_rate": float(INITIAL_LEARNING_RATE),
    "weight_decay": float(WEIGHT_DECAY),
    "maximum_gradient_norm": float(MAXIMUM_GRADIENT_NORM),
    "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
    "scheduler_patience": int(SCHEDULER_PATIENCE),
    "scheduler_reduction_factor": float(SCHEDULER_REDUCTION_FACTOR),
    "minimum_learning_rate": float(MINIMUM_LEARNING_RATE),
    "classification_threshold": float(CLASSIFICATION_THRESHOLD),
    "modality_dropout_probability": float(
        complete_model.modality_dropout_probability
    ),
    "embedding_dimension": int(MODALITY_EMBEDDING_DIM),
    "mri_final_channels": 256,
    "mri_patch_embedding_dimension": 256,
    "mri_transformer_heads": 8,
    "mri_transformer_layers": 1,
    "mri_transformer_feedforward_dimension": 512,
    "three_mt_attention_heads": 4,
    "evidential_head_hidden_dimension": 32,
    "number_of_classes": int(NUMBER_OF_CLASSES),
    "class_order": list(PROGNOSIS_CLASS_ORDER),
    "modality_order": list(MODALITY_ORDER),
    "three_mt_cascade_order": list(THREE_MT_CASCADE_ORDER),
    "trainable_parameters": int(count_trainable_parameters(complete_model)),
    "device": str(DEVICE),
    "cuda_device_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "project_root": str(PROJECT_ROOT),
    "selected_input_path": str(SELECTED_INPUT_PATH),
}

with TRAINING_CONFIGURATION_PATH.open("w", encoding="utf-8") as f:
    json.dump(training_configuration, f, indent=2)

print("=" * 72)
print("TRAINING CONFIGURATION")
print("=" * 72)
print(f"Architecture: {ARCHITECTURE_PRESET}")
print(f"Trainable parameters: {count_trainable_parameters(complete_model):,}")
print(f"Shared embedding: {MODALITY_EMBEDDING_DIM}")
print("MRI channels: 16 -> 32 -> 64 -> 128 -> 256")
print("MRI patch dim / heads / FF: 256 / 8 / 512")
print("3MT attention heads: 4")
print(f"Maximum epochs: {MAXIMUM_EPOCHS}")
print(f"Output directory: {FOLD_TRAINING_DIR}")


# 18. Defining reusable training and validation epoch functions

from collections import defaultdict


# Initial numerical-precision policy

# Train in full float32 precision.
#
# This is computationally feasible on the available A100 GPU and
# avoids introducing mixed-precision instability into the Dirichlet
# digamma, log-gamma, and Dempster-Shafer calculations.
USE_MIXED_PRECISION = False


# Recursively move a nested batch to the selected device

def move_nested_batch_to_device(
    value,
    device,
):
    """
    Recursively move tensors in dictionaries, lists, and tuples
    to the selected PyTorch device.

    Non-tensor values are preserved unchanged.
    """

    if isinstance(
        value,
        torch.Tensor,
    ):
        return value.to(
            device,
            non_blocking=True,
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: move_nested_batch_to_device(
                nested_value,
                device,
            )
            for key, nested_value in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            move_nested_batch_to_device(
                nested_value,
                device,
            )
            for nested_value in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            move_nested_batch_to_device(
                nested_value,
                device,
            )
            for nested_value in value
        )

    return value


# Convert one completed forward pass into stored predictions

def extract_batch_prediction_arrays(
    batch,
    model_output,
):
    """
    Extract participant-level targets, predictions, uncertainty,
    pathway outputs, and modality counts from one mini-batch.
    """

    final_output = model_output[
        "final_output"
    ]

    joint_opinion = model_output[
        "joint_opinion"
    ]

    tmc_output = model_output[
        "tmc_output"
    ]


    extracted = {
        "rid":
            batch["rid"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1),

        "target":
            batch["target"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1),

        "final_p_pMCI":
            final_output[
                "probabilities"
            ][
                :,
                1,
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "final_uncertainty":
            final_output[
                "uncertainty"
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "three_mt_weight":
            final_output[
                "three_mt_weight"
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "tmc_weight":
            final_output[
                "tmc_weight"
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "three_mt_p_pMCI":
            joint_opinion[
                "probabilities"
            ][
                :,
                1,
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "three_mt_uncertainty":
            joint_opinion[
                "uncertainty"
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "tmc_p_pMCI":
            tmc_output[
                "probabilities"
            ][
                :,
                1,
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "tmc_uncertainty":
            tmc_output[
                "uncertainty"
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "original_modality_count":
            model_output[
                "original_branch_masks"
            ]
            .sum(
                dim=-1
            )
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),

        "effective_modality_count":
            model_output[
                "effective_branch_masks"
            ]
            .sum(
                dim=-1
            )
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1),
    }

    return extracted


# Concatenate the participant outputs collected across an epoch

def concatenate_epoch_prediction_storage(
    prediction_storage,
):
    """
    Concatenate a dictionary of mini-batch NumPy arrays.
    """

    concatenated = {}

    for key, value_list in prediction_storage.items():

        if len(value_list) == 0:
            concatenated[key] = np.asarray([])

        else:
            concatenated[key] = np.concatenate(
                value_list,
                axis=0,
            )

    return concatenated


# Convert stored predictions into a participant-level table

def build_epoch_prediction_table(
    concatenated_predictions,
    split_name,
    epoch,
):
    """
    Build one participant-level DataFrame for an epoch.
    """

    prediction_table = pd.DataFrame(
        {
            "RID":
                concatenated_predictions[
                    "rid"
                ].astype(
                    np.int64
                ),

            "TARGET":
                concatenated_predictions[
                    "target"
                ].astype(
                    np.int64
                ),

            "FINAL_P_pMCI":
                concatenated_predictions[
                    "final_p_pMCI"
                ],

            "FINAL_UNCERTAINTY":
                concatenated_predictions[
                    "final_uncertainty"
                ],

            "W_3MT":
                concatenated_predictions[
                    "three_mt_weight"
                ],

            "W_TMC":
                concatenated_predictions[
                    "tmc_weight"
                ],

            "THREE_MT_P_pMCI":
                concatenated_predictions[
                    "three_mt_p_pMCI"
                ],

            "THREE_MT_UNCERTAINTY":
                concatenated_predictions[
                    "three_mt_uncertainty"
                ],

            "TMC_P_pMCI":
                concatenated_predictions[
                    "tmc_p_pMCI"
                ],

            "TMC_UNCERTAINTY":
                concatenated_predictions[
                    "tmc_uncertainty"
                ],

            "ORIGINAL_MODALITY_COUNT":
                concatenated_predictions[
                    "original_modality_count"
                ].astype(
                    np.int64
                ),

            "EFFECTIVE_MODALITY_COUNT":
                concatenated_predictions[
                    "effective_modality_count"
                ].astype(
                    np.int64
                ),
        }
    )

    prediction_table.insert(
        loc=0,
        column="EPOCH",
        value=int(epoch),
    )

    prediction_table.insert(
        loc=1,
        column="SPLIT",
        value=str(split_name),
    )

    prediction_table[
        "FINAL_PREDICTED_CLASS"
    ] = (
        prediction_table[
            "FINAL_P_pMCI"
        ]
        >= CLASSIFICATION_THRESHOLD
    ).astype(
        np.int64
    )

    return prediction_table


# Metric helpers preserved from the original fixed-fusion notebook

def calculate_binary_expected_calibration_error(
    targets,
    positive_class_probabilities,
    number_of_bins=10,
):
    """
    Calculate equal-width binary expected calibration error.

    Confidence is the probability assigned to the predicted class.
    """

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    positive_class_probabilities = np.asarray(
        positive_class_probabilities,
        dtype=np.float64,
    )

    if targets.size == 0:
        return float("nan")

    predicted_classes = (
        positive_class_probabilities
        >= 0.5
    ).astype(
        np.int64
    )

    predicted_confidences = np.where(
        predicted_classes == 1,
        positive_class_probabilities,
        1.0 - positive_class_probabilities,
    )

    prediction_correctness = (
        predicted_classes
        == targets
    ).astype(
        np.float64
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    expected_calibration_error = 0.0

    sample_count = targets.size

    for bin_index in range(
        number_of_bins
    ):

        lower_edge = bin_edges[
            bin_index
        ]

        upper_edge = bin_edges[
            bin_index + 1
        ]

        if bin_index == 0:

            in_bin = (
                predicted_confidences
                >= lower_edge
            ) & (
                predicted_confidences
                <= upper_edge
            )

        else:

            in_bin = (
                predicted_confidences
                > lower_edge
            ) & (
                predicted_confidences
                <= upper_edge
            )

        bin_count = int(
            in_bin.sum()
        )

        if bin_count == 0:
            continue

        mean_confidence = float(
            predicted_confidences[
                in_bin
            ].mean()
        )

        mean_accuracy = float(
            prediction_correctness[
                in_bin
            ].mean()
        )

        expected_calibration_error += (
            bin_count
            / sample_count
        ) * abs(
            mean_accuracy
            - mean_confidence
        )

    return float(
        expected_calibration_error
    )


# Safe metric helpers

def safely_calculate_roc_auc(
    targets,
    probabilities,
):
    """
    Return NaN when ROC AUC is undefined because only one class
    is present in the supplied targets.
    """

    if np.unique(targets).size < 2:
        return float("nan")

    return float(
        roc_auc_score(
            targets,
            probabilities,
        )
    )


def safely_calculate_average_precision(
    targets,
    probabilities,
):
    """
    Return NaN when average precision is not meaningful because
    the supplied targets contain no positive examples.
    """

    if np.sum(targets == 1) == 0:
        return float("nan")

    return float(
        average_precision_score(
            targets,
            probabilities,
        )
    )


# Complete binary prognosis metrics

def calculate_binary_classification_metrics(
    targets,
    positive_class_probabilities,
    uncertainties=None,
    three_mt_weights=None,
    classification_threshold=0.50,
):
    """
    Calculate discrimination, classification, calibration, and
    uncertainty summaries for the pMCI-positive prognosis task.
    """

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    positive_class_probabilities = np.asarray(
        positive_class_probabilities,
        dtype=np.float64,
    )

    if targets.size == 0:
        raise ValueError(
            "At least one target is required to calculate metrics."
        )

    if (
        targets.shape[0]
        != positive_class_probabilities.shape[0]
    ):
        raise ValueError(
            "Targets and probabilities must contain the same "
            "number of participants."
        )

    positive_class_probabilities = np.clip(
        positive_class_probabilities,
        0.0,
        1.0,
    )

    predicted_classes = (
        positive_class_probabilities
        >= classification_threshold
    ).astype(
        np.int64
    )

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        targets,
        predicted_classes,
        labels=[0, 1],
    ).ravel()


    sensitivity_denominator = (
        true_positive
        + false_negative
    )

    specificity_denominator = (
        true_negative
        + false_positive
    )

    sensitivity = (
        true_positive
        / sensitivity_denominator
        if sensitivity_denominator > 0
        else float("nan")
    )

    specificity = (
        true_negative
        / specificity_denominator
        if specificity_denominator > 0
        else float("nan")
    )


    clipped_probabilities = np.clip(
        positive_class_probabilities,
        1e-7,
        1.0 - 1e-7,
    )


    metrics = {
        "roc_auc":
            safely_calculate_roc_auc(
                targets,
                positive_class_probabilities,
            ),

        "average_precision":
            safely_calculate_average_precision(
                targets,
                positive_class_probabilities,
            ),

        "accuracy":
            float(
                accuracy_score(
                    targets,
                    predicted_classes,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    targets,
                    predicted_classes,
                )
            ),

        "sensitivity":
            float(
                sensitivity
            ),

        "specificity":
            float(
                specificity
            ),

        "precision":
            float(
                precision_score(
                    targets,
                    predicted_classes,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    targets,
                    predicted_classes,
                    zero_division=0,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    targets,
                    positive_class_probabilities,
                )
            ),

        "negative_log_likelihood":
            float(
                log_loss(
                    targets,
                    np.column_stack(
                        [
                            1.0
                            - clipped_probabilities,

                            clipped_probabilities,
                        ]
                    ),
                    labels=[0, 1],
                )
            ),

        "expected_calibration_error":
            calculate_binary_expected_calibration_error(
                targets=targets,

                positive_class_probabilities=
                    positive_class_probabilities,

                number_of_bins=10,
            ),

        "classification_threshold":
            float(
                classification_threshold
            ),

        "true_negative":
            int(
                true_negative
            ),

        "false_positive":
            int(
                false_positive
            ),

        "false_negative":
            int(
                false_negative
            ),

        "true_positive":
            int(
                true_positive
            ),
    }


    if uncertainties is not None:

        uncertainties = np.asarray(
            uncertainties,
            dtype=np.float64,
        )

        metrics[
            "mean_uncertainty"
        ] = float(
            uncertainties.mean()
        )

        metrics[
            "std_uncertainty"
        ] = float(
            uncertainties.std()
        )


    if three_mt_weights is not None:

        three_mt_weights = np.asarray(
            three_mt_weights,
            dtype=np.float64,
        )

        metrics[
            "mean_three_mt_weight"
        ] = float(
            three_mt_weights.mean()
        )

        metrics[
            "std_three_mt_weight"
        ] = float(
            three_mt_weights.std()
        )

        metrics[
            "minimum_three_mt_weight"
        ] = float(
            three_mt_weights.min()
        )

        metrics[
            "maximum_three_mt_weight"
        ] = float(
            three_mt_weights.max()
        )


    return metrics

# Calculate metrics for all three prediction outputs

def calculate_epoch_prediction_metrics(
    concatenated_predictions,
):
    """
    Calculate metrics for:

    1. the final fixed-equal hybrid output;
    2. the 3MT-only joint output;
    3. the TMC-only fused output.
    """

    targets = concatenated_predictions[
        "target"
    ]

    final_metrics = (
        calculate_binary_classification_metrics(
            targets=targets,

            positive_class_probabilities=
                concatenated_predictions[
                    "final_p_pMCI"
                ],

            uncertainties=
                concatenated_predictions[
                    "final_uncertainty"
                ],

            three_mt_weights=
                concatenated_predictions[
                    "three_mt_weight"
                ],

            classification_threshold=
                CLASSIFICATION_THRESHOLD,
        )
    )

    three_mt_metrics = (
        calculate_binary_classification_metrics(
            targets=targets,

            positive_class_probabilities=
                concatenated_predictions[
                    "three_mt_p_pMCI"
                ],

            uncertainties=
                concatenated_predictions[
                    "three_mt_uncertainty"
                ],

            classification_threshold=
                CLASSIFICATION_THRESHOLD,
        )
    )

    tmc_metrics = (
        calculate_binary_classification_metrics(
            targets=targets,

            positive_class_probabilities=
                concatenated_predictions[
                    "tmc_p_pMCI"
                ],

            uncertainties=
                concatenated_predictions[
                    "tmc_uncertainty"
                ],

            classification_threshold=
                CLASSIFICATION_THRESHOLD,
        )
    )

    return {
        "final": final_metrics,
        "three_mt": three_mt_metrics,
        "tmc": tmc_metrics,
    }


# Initialise the loss accumulator used within an epoch

EPOCH_LOSS_NAMES = [
    "total_loss",
    "final_loss",
    "joint_loss",
    "tmc_loss",
    "modality_loss",
    "auxiliary_loss",
    "raw_gate_loss",
    "annealed_gate_loss",
]


def initialise_epoch_loss_storage():
    """
    Create participant-weighted loss totals.
    """

    return {
        loss_name: 0.0
        for loss_name in EPOCH_LOSS_NAMES
    }


def update_epoch_loss_storage(
    storage,
    loss_output,
    batch_size,
):
    """
    Add one batch's losses, weighted by the number of participants.
    """

    for loss_name in EPOCH_LOSS_NAMES:

        storage[loss_name] += (
            float(
                loss_output[
                    loss_name
                ]
                .detach()
                .float()
                .item()
            )
            * batch_size
        )


def finalise_epoch_loss_storage(
    storage,
    participant_count,
):
    """
    Convert accumulated loss sums into participant-weighted means.
    """

    if participant_count <= 0:
        raise ValueError(
            "The epoch contained no participants."
        )

    return {
        loss_name:
            loss_sum
            / participant_count

        for loss_name, loss_sum in storage.items()
    }


# Run one complete training epoch

def run_training_epoch(
    model,
    data_loader,
    objective,
    optimizer,
    device,
    epoch,
    maximum_gradient_norm,
):
    """
    Train the complete model for one epoch.
    """

    model.train()

    loss_storage = (
        initialise_epoch_loss_storage()
    )

    prediction_storage = defaultdict(
        list
    )

    participant_count = 0

    batch_gradient_norms = []

    effective_modality_counts = []


    for batch_index, batch in enumerate(
        data_loader
    ):

        batch = move_nested_batch_to_device(
            batch,
            device,
        )

        targets = batch[
            "target"
        ].long()

        batch_size = targets.shape[0]

        participant_count += batch_size


        # Clear gradients from the previous mini-batch

        optimizer.zero_grad(
            set_to_none=True
        )


        # Complete forward pass

        model_output = model(
            modalities=batch[
                "modalities"
            ],

            original_branch_masks=batch[
                "branch_masks"
            ],
        )


        # Complete multi-output objective

        loss_output = objective(
            model_output=model_output,
            targets=targets,
            epoch=epoch,
        )

        total_loss = loss_output[
            "total_loss"
        ]


        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "A non-finite training loss was encountered "
                f"at epoch {epoch}, batch {batch_index}."
            )


        # Backpropagation

        total_loss.backward()


        # Global gradient clipping

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=maximum_gradient_norm,
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):
            raise FloatingPointError(
                "A non-finite gradient norm was encountered "
                f"at epoch {epoch}, batch {batch_index}."
            )

        batch_gradient_norms.append(
            float(
                gradient_norm.detach().item()
            )
        )


        # Parameter update

        optimizer.step()


        # Accumulate losses and predictions

        update_epoch_loss_storage(
            storage=loss_storage,
            loss_output=loss_output,
            batch_size=batch_size,
        )

        extracted_predictions = (
            extract_batch_prediction_arrays(
                batch=batch,
                model_output=model_output,
            )
        )

        for key, values in (
            extracted_predictions.items()
        ):
            prediction_storage[key].append(
                values
            )

        effective_modality_counts.extend(
            extracted_predictions[
                "effective_modality_count"
            ].tolist()
        )


    # Finalise the complete training epoch

    mean_losses = finalise_epoch_loss_storage(
        storage=loss_storage,
        participant_count=participant_count,
    )

    concatenated_predictions = (
        concatenate_epoch_prediction_storage(
            prediction_storage
        )
    )

    prediction_metrics = (
        calculate_epoch_prediction_metrics(
            concatenated_predictions
        )
    )

    prediction_table = (
        build_epoch_prediction_table(
            concatenated_predictions=
                concatenated_predictions,

            split_name="train",

            epoch=epoch,
        )
    )


    gradient_summary = {
        "mean_gradient_norm":
            float(
                np.mean(
                    batch_gradient_norms
                )
            ),

        "maximum_gradient_norm_before_clipping":
            float(
                np.max(
                    batch_gradient_norms
                )
            ),

        "minimum_gradient_norm":
            float(
                np.min(
                    batch_gradient_norms
                )
            ),
    }


    modality_dropout_summary = {
        "mean_effective_modality_count":
            float(
                np.mean(
                    effective_modality_counts
                )
            ),

        "minimum_effective_modality_count":
            float(
                np.min(
                    effective_modality_counts
                )
            ),

        "maximum_effective_modality_count":
            float(
                np.max(
                    effective_modality_counts
                )
            ),
    }


    return {
        "losses":
            mean_losses,

        "metrics":
            prediction_metrics,

        "predictions":
            prediction_table,

        "gradient_summary":
            gradient_summary,

        "modality_dropout_summary":
            modality_dropout_summary,

        "participant_count":
            int(
                participant_count
            ),

        "batch_count":
            int(
                len(data_loader)
            ),
    }


# Run one complete validation epoch

def run_validation_epoch(
    model,
    data_loader,
    objective,
    device,
    epoch,
):
    """
    Evaluate the complete model for one epoch without gradients
    or training-time modality dropout.
    """

    model.eval()

    loss_storage = (
        initialise_epoch_loss_storage()
    )

    prediction_storage = defaultdict(
        list
    )

    participant_count = 0


    with torch.no_grad():

        for batch_index, batch in enumerate(
            data_loader
        ):

            batch = move_nested_batch_to_device(
                batch,
                device,
            )

            targets = batch[
                "target"
            ].long()

            batch_size = targets.shape[0]

            participant_count += batch_size


            # Complete evaluation forward pass

            model_output = model(
                modalities=batch[
                    "modalities"
                ],

                original_branch_masks=batch[
                    "branch_masks"
                ],
            )


            # Validation objective

            loss_output = objective(
                model_output=model_output,
                targets=targets,
                epoch=epoch,
            )

            if not torch.isfinite(
                loss_output[
                    "total_loss"
                ]
            ):
                raise FloatingPointError(
                    "A non-finite validation loss was encountered "
                    f"at epoch {epoch}, batch {batch_index}."
                )


            # Accumulate losses and predictions

            update_epoch_loss_storage(
                storage=loss_storage,
                loss_output=loss_output,
                batch_size=batch_size,
            )

            extracted_predictions = (
                extract_batch_prediction_arrays(
                    batch=batch,
                    model_output=model_output,
                )
            )

            for key, values in (
                extracted_predictions.items()
            ):
                prediction_storage[key].append(
                    values
                )


    # Finalise the complete validation epoch

    mean_losses = finalise_epoch_loss_storage(
        storage=loss_storage,
        participant_count=participant_count,
    )

    concatenated_predictions = (
        concatenate_epoch_prediction_storage(
            prediction_storage
        )
    )

    prediction_metrics = (
        calculate_epoch_prediction_metrics(
            concatenated_predictions
        )
    )

    prediction_table = (
        build_epoch_prediction_table(
            concatenated_predictions=
                concatenated_predictions,

            split_name="validation",

            epoch=epoch,
        )
    )


    original_modality_counts = (
        concatenated_predictions[
            "original_modality_count"
        ]
    )

    effective_modality_counts = (
        concatenated_predictions[
            "effective_modality_count"
        ]
    )

    maximum_mask_count_difference = float(
        np.max(
            np.abs(
                original_modality_counts
                - effective_modality_counts
            )
        )
    )


    return {
        "losses":
            mean_losses,

        "metrics":
            prediction_metrics,

        "predictions":
            prediction_table,

        "participant_count":
            int(
                participant_count
            ),

        "batch_count":
            int(
                len(data_loader)
            ),

        "maximum_evaluation_modality_count_difference":
            maximum_mask_count_difference,
    }


# Display the configured epoch-function summary

print("=" * 72)
print("TRAINING AND VALIDATION EPOCH FUNCTIONS")
print("=" * 72)

print(
    f"\nMixed precision enabled: "
    f"{USE_MIXED_PRECISION}"
)

print(
    "Training batches per epoch: "
    f"{len(train_loader)}"
)

print(
    "Validation batches per epoch: "
    f"{len(validation_loader)}"
)

print(
    "Training participants: "
    f"{len(train_loader.dataset)}"
)

print(
    "Validation participants: "
    f"{len(validation_loader.dataset)}"
)

print(
    "\nTraining epoch operations:"
)

print(
    "- forward pass with modality dropout;"
)

print(
    "- complete joint loss calculation;"
)

print(
    "- backpropagation;"
)

print(
    "- global gradient clipping;"
)

print(
    "- AdamW parameter update;"
)

print(
    "- prediction and uncertainty accumulation."
)

print(
    "\nValidation epoch operations:"
)

print(
    "- evaluation mode;"
)

print(
    "- no modality dropout;"
)

print(
    "- no gradient calculation;"
)

print(
    "- complete validation loss and metric calculation."
)

print(
    "\nThe epoch functions are defined."
)

print(
    "No complete training or validation epoch has been run yet."
)

print(
    "The next step will create the checkpointed multi-epoch "
    "training loop and begin model optimisation."
)

# 19. Training with live batch and epoch progress

import time
import traceback
from collections import defaultdict
from datetime import datetime

from tqdm.auto import tqdm


# Resume and progress-display behaviour

# The formal first run begins from epoch 1.
# Resume is enabled only when FOLD_RUN_MODE was explicitly set
# to "resume" for this experiment and fold.
RESUME_FROM_LAST_CHECKPOINT = (
    FOLD_RUN_MODE == "resume"
)

# Refresh the live progress display after every batch.
PROGRESS_UPDATE_INTERVAL = 1


# Loss-configuration serialisation

def obtain_training_objective_configuration(
    objective,
):
    """
    Return the principal loss settings in a checkpoint-safe form.
    """

    return {
        "evidential_annealing_epochs": int(
            objective.evidential_annealing_epochs
        ),

        "gate_regularisation_epochs": int(
            objective.gate_regularisation_epochs
        ),

        "joint_loss_weight": float(
            objective.joint_loss_weight
        ),

        "tmc_loss_weight": float(
            objective.tmc_loss_weight
        ),

        "modality_loss_weight": float(
            objective.modality_loss_weight
        ),

        "auxiliary_loss_weight": float(
            objective.auxiliary_loss_weight
        ),

        "gate_loss_weight": float(
            objective.gate_loss_weight
        ),

        "class_weights": (
            None
            if objective.class_weights is None
            else
            objective.class_weights
            .detach()
            .cpu()
            .tolist()
        ),
    }


# Flatten one epoch into one history row

def build_training_history_row(
    epoch,
    learning_rate,
    epoch_duration_seconds,
    training_result,
    validation_result,
    best_validation_auc,
    epochs_without_improvement,
    checkpoint_improved,
):
    """
    Create one flat row containing losses, metrics, optimisation
    diagnostics, and checkpoint information.
    """

    history_row = {
        "epoch":
            int(epoch),

        "learning_rate":
            float(learning_rate),

        "epoch_duration_seconds":
            float(epoch_duration_seconds),

        "best_validation_auc":
            float(best_validation_auc),

        "epochs_without_improvement":
            int(epochs_without_improvement),

        "checkpoint_improved":
            bool(checkpoint_improved),

        "train_participants":
            int(
                training_result[
                    "participant_count"
                ]
            ),

        "validation_participants":
            int(
                validation_result[
                    "participant_count"
                ]
            ),

        "train_batches":
            int(
                training_result[
                    "batch_count"
                ]
            ),

        "validation_batches":
            int(
                validation_result[
                    "batch_count"
                ]
            ),

        "train_mean_gradient_norm":
            float(
                training_result[
                    "gradient_summary"
                ]["mean_gradient_norm"]
            ),

        "train_maximum_gradient_norm_before_clipping":
            float(
                training_result[
                    "gradient_summary"
                ][
                    "maximum_gradient_norm_before_clipping"
                ]
            ),

        "train_mean_effective_modality_count":
            float(
                training_result[
                    "modality_dropout_summary"
                ][
                    "mean_effective_modality_count"
                ]
            ),

        "train_minimum_effective_modality_count":
            float(
                training_result[
                    "modality_dropout_summary"
                ][
                    "minimum_effective_modality_count"
                ]
            ),

        "train_maximum_effective_modality_count":
            float(
                training_result[
                    "modality_dropout_summary"
                ][
                    "maximum_effective_modality_count"
                ]
            ),

        "validation_maximum_modality_count_difference":
            float(
                validation_result[
                    "maximum_evaluation_modality_count_difference"
                ]
            ),
    }


    # Add all training and validation losses

    for loss_name, loss_value in (
        training_result[
            "losses"
        ].items()
    ):
        history_row[
            f"train_{loss_name}"
        ] = float(
            loss_value
        )

    for loss_name, loss_value in (
        validation_result[
            "losses"
        ].items()
    ):
        history_row[
            f"validation_{loss_name}"
        ] = float(
            loss_value
        )


    # Add metrics for all three prediction outputs

    for output_name in [
        "final",
        "three_mt",
        "tmc",
    ]:

        for metric_name, metric_value in (
            training_result[
                "metrics"
            ][output_name].items()
        ):
            history_row[
                f"train_{output_name}_{metric_name}"
            ] = metric_value

        for metric_name, metric_value in (
            validation_result[
                "metrics"
            ][output_name].items()
        ):
            history_row[
                f"validation_{output_name}_{metric_name}"
            ] = metric_value


    return history_row


# Save a fully recoverable checkpoint

def save_training_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_validation_auc,
    epochs_without_improvement,
    validation_metrics,
    training_history,
):
    """
    Save the state required to reproduce or continue training.
    """

    checkpoint = {
        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_validation_auc":
            float(best_validation_auc),

        "epochs_without_improvement":
            int(epochs_without_improvement),

        "validation_metrics":
            validation_metrics,

        "training_configuration":
            training_configuration,

        "training_objective_configuration":
            obtain_training_objective_configuration(
                training_objective
            ),

        "training_history":
            training_history,

        "random_seed":
            int(GLOBAL_RANDOM_SEED),

        "task":
            SELECTED_TASK,

        "fold":
            int(SELECTED_FOLD),

        "saved_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# GPU-memory helper for the progress display

def current_cuda_memory_gb():
    """
    Return currently allocated CUDA memory in gigabytes.
    """

    if not torch.cuda.is_available():
        return 0.0

    return float(
        torch.cuda.memory_allocated(
            DEVICE
        )
        / (1024 ** 3)
    )


# One training epoch with a live batch progress bar

def run_training_epoch_with_progress(
    model,
    data_loader,
    objective,
    optimizer,
    device,
    epoch,
    maximum_epochs,
    maximum_gradient_norm,
):
    """
    Train for one epoch while displaying batch-level progress.
    """

    model.train()

    loss_storage = (
        initialise_epoch_loss_storage()
    )

    prediction_storage = defaultdict(
        list
    )

    participant_count = 0

    batch_gradient_norms = []

    effective_modality_counts = []

    running_total_loss_sum = 0.0

    phase_start_time = time.time()


    progress_bar = tqdm(
        enumerate(
            data_loader,
            start=1,
        ),
        total=len(
            data_loader
        ),
        desc=(
            f"Epoch {epoch:02d}/{maximum_epochs} | training"
        ),
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )


    for batch_number, batch in progress_bar:

        batch = move_nested_batch_to_device(
            batch,
            device,
        )

        targets = batch[
            "target"
        ].long()

        batch_size = int(
            targets.shape[0]
        )

        participant_count += (
            batch_size
        )


        # Forward pass and loss

        optimizer.zero_grad(
            set_to_none=True
        )

        model_output = model(
            modalities=batch[
                "modalities"
            ],

            original_branch_masks=batch[
                "branch_masks"
            ],
        )

        loss_output = objective(
            model_output=model_output,
            targets=targets,
            epoch=epoch,
        )

        total_loss = loss_output[
            "total_loss"
        ]


        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "A non-finite training loss was encountered "
                f"at epoch {epoch}, batch {batch_number}."
            )


        # Backpropagation, clipping, and update

        total_loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=maximum_gradient_norm,
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):
            raise FloatingPointError(
                "A non-finite gradient norm was encountered "
                f"at epoch {epoch}, batch {batch_number}."
            )

        optimizer.step()


        # Accumulate diagnostics

        current_total_loss = float(
            total_loss.detach().item()
        )

        running_total_loss_sum += (
            current_total_loss
            * batch_size
        )

        running_mean_total_loss = (
            running_total_loss_sum
            / participant_count
        )

        batch_gradient_norms.append(
            float(
                gradient_norm.detach().item()
            )
        )

        update_epoch_loss_storage(
            storage=loss_storage,
            loss_output=loss_output,
            batch_size=batch_size,
        )

        extracted_predictions = (
            extract_batch_prediction_arrays(
                batch=batch,
                model_output=model_output,
            )
        )

        for key, values in (
            extracted_predictions.items()
        ):
            prediction_storage[
                key
            ].append(
                values
            )

        effective_modality_counts.extend(
            extracted_predictions[
                "effective_modality_count"
            ].tolist()
        )


        # Update the visible progress information

        if (
            batch_number
            % PROGRESS_UPDATE_INTERVAL
            == 0
            or batch_number
            == len(data_loader)
        ):

            elapsed_minutes = (
                time.time()
                - phase_start_time
            ) / 60.0

            progress_bar.set_postfix(
                {
                    "loss":
                        f"{current_total_loss:.3f}",

                    "avg":
                        f"{running_mean_total_loss:.3f}",

                    "grad":
                        f"{float(gradient_norm):.2f}",

                    "lr":
                        f"{optimizer.param_groups[0]['lr']:.1e}",

                    "GPU":
                        f"{current_cuda_memory_gb():.1f}GB",

                    "elapsed":
                        f"{elapsed_minutes:.1f}m",
                },
                refresh=True,
            )


    # Finalise the training epoch

    mean_losses = finalise_epoch_loss_storage(
        storage=loss_storage,
        participant_count=participant_count,
    )

    concatenated_predictions = (
        concatenate_epoch_prediction_storage(
            prediction_storage
        )
    )

    prediction_metrics = (
        calculate_epoch_prediction_metrics(
            concatenated_predictions
        )
    )

    prediction_table = (
        build_epoch_prediction_table(
            concatenated_predictions=
                concatenated_predictions,

            split_name="train",

            epoch=epoch,
        )
    )


    gradient_summary = {
        "mean_gradient_norm":
            float(
                np.mean(
                    batch_gradient_norms
                )
            ),

        "maximum_gradient_norm_before_clipping":
            float(
                np.max(
                    batch_gradient_norms
                )
            ),

        "minimum_gradient_norm":
            float(
                np.min(
                    batch_gradient_norms
                )
            ),
    }


    modality_dropout_summary = {
        "mean_effective_modality_count":
            float(
                np.mean(
                    effective_modality_counts
                )
            ),

        "minimum_effective_modality_count":
            float(
                np.min(
                    effective_modality_counts
                )
            ),

        "maximum_effective_modality_count":
            float(
                np.max(
                    effective_modality_counts
                )
            ),
    }


    return {
        "losses":
            mean_losses,

        "metrics":
            prediction_metrics,

        "predictions":
            prediction_table,

        "gradient_summary":
            gradient_summary,

        "modality_dropout_summary":
            modality_dropout_summary,

        "participant_count":
            int(
                participant_count
            ),

        "batch_count":
            int(
                len(data_loader)
            ),
    }


# One validation epoch with a live batch progress bar

def run_validation_epoch_with_progress(
    model,
    data_loader,
    objective,
    device,
    epoch,
    maximum_epochs,
):
    """
    Validate for one epoch while displaying batch-level progress.
    """

    model.eval()

    loss_storage = (
        initialise_epoch_loss_storage()
    )

    prediction_storage = defaultdict(
        list
    )

    participant_count = 0

    running_total_loss_sum = 0.0

    phase_start_time = time.time()


    progress_bar = tqdm(
        enumerate(
            data_loader,
            start=1,
        ),
        total=len(
            data_loader
        ),
        desc=(
            f"Epoch {epoch:02d}/{maximum_epochs} | validation"
        ),
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )


    with torch.no_grad():

        for batch_number, batch in progress_bar:

            batch = move_nested_batch_to_device(
                batch,
                device,
            )

            targets = batch[
                "target"
            ].long()

            batch_size = int(
                targets.shape[0]
            )

            participant_count += (
                batch_size
            )


            # Forward pass and validation loss

            model_output = model(
                modalities=batch[
                    "modalities"
                ],

                original_branch_masks=batch[
                    "branch_masks"
                ],
            )

            loss_output = objective(
                model_output=model_output,
                targets=targets,
                epoch=epoch,
            )

            total_loss = loss_output[
                "total_loss"
            ]


            if not torch.isfinite(
                total_loss
            ):
                raise FloatingPointError(
                    "A non-finite validation loss was encountered "
                    f"at epoch {epoch}, batch {batch_number}."
                )


            # Accumulate diagnostics and predictions

            current_total_loss = float(
                total_loss.detach().item()
            )

            running_total_loss_sum += (
                current_total_loss
                * batch_size
            )

            running_mean_total_loss = (
                running_total_loss_sum
                / participant_count
            )

            update_epoch_loss_storage(
                storage=loss_storage,
                loss_output=loss_output,
                batch_size=batch_size,
            )

            extracted_predictions = (
                extract_batch_prediction_arrays(
                    batch=batch,
                    model_output=model_output,
                )
            )

            for key, values in (
                extracted_predictions.items()
            ):
                prediction_storage[
                    key
                ].append(
                    values
                )


            # Update the visible progress information

            if (
                batch_number
                % PROGRESS_UPDATE_INTERVAL
                == 0
                or batch_number
                == len(data_loader)
            ):

                elapsed_minutes = (
                    time.time()
                    - phase_start_time
                ) / 60.0

                progress_bar.set_postfix(
                    {
                        "loss":
                            f"{current_total_loss:.3f}",

                        "avg":
                            f"{running_mean_total_loss:.3f}",

                        "GPU":
                            f"{current_cuda_memory_gb():.1f}GB",

                        "elapsed":
                            f"{elapsed_minutes:.1f}m",
                    },
                    refresh=True,
                )


    # Finalise the validation epoch

    mean_losses = finalise_epoch_loss_storage(
        storage=loss_storage,
        participant_count=participant_count,
    )

    concatenated_predictions = (
        concatenate_epoch_prediction_storage(
            prediction_storage
        )
    )

    prediction_metrics = (
        calculate_epoch_prediction_metrics(
            concatenated_predictions
        )
    )

    prediction_table = (
        build_epoch_prediction_table(
            concatenated_predictions=
                concatenated_predictions,

            split_name="validation",

            epoch=epoch,
        )
    )


    maximum_mask_count_difference = float(
        np.max(
            np.abs(
                concatenated_predictions[
                    "original_modality_count"
                ]
                -
                concatenated_predictions[
                    "effective_modality_count"
                ]
            )
        )
    )


    return {
        "losses":
            mean_losses,

        "metrics":
            prediction_metrics,

        "predictions":
            prediction_table,

        "participant_count":
            int(
                participant_count
            ),

        "batch_count":
            int(
                len(data_loader)
            ),

        "maximum_evaluation_modality_count_difference":
            maximum_mask_count_difference,
    }


# Initial training state

training_history = []

starting_epoch = 1

best_validation_auc = float(
    "-inf"
)

epochs_without_improvement = 0


# Resume from the latest fully completed epoch

if (
    RESUME_FROM_LAST_CHECKPOINT
    and LAST_CHECKPOINT_PATH.exists()
):

    print(
        "Loading the latest fully completed checkpoint:",
        flush=True,
    )

    print(
        LAST_CHECKPOINT_PATH,
        flush=True,
    )

    resumed_checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    complete_model.load_state_dict(
        resumed_checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        resumed_checkpoint[
            "optimizer_state_dict"
        ]
    )

    learning_rate_scheduler.load_state_dict(
        resumed_checkpoint[
            "scheduler_state_dict"
        ]
    )

    completed_epoch = int(
        resumed_checkpoint[
            "epoch"
        ]
    )

    starting_epoch = (
        completed_epoch
        + 1
    )

    best_validation_auc = float(
        resumed_checkpoint[
            "best_validation_auc"
        ]
    )

    epochs_without_improvement = int(
        resumed_checkpoint[
            "epochs_without_improvement"
        ]
    )

    training_history = list(
        resumed_checkpoint.get(
            "training_history",
            [],
        )
    )

    print(
        f"\nResuming after epoch {completed_epoch}.",
        flush=True,
    )

    print(
        f"Next epoch: {starting_epoch}",
        flush=True,
    )

    print(
        "Best validation ROC AUC so far: "
        f"{best_validation_auc:.6f}",
        flush=True,
    )

else:

    print(
        "No fully completed checkpoint was loaded.",
        flush=True,
    )

    print(
        "Fresh training begins from the newly initialised model state.",
        flush=True,
    )


# Main multi-epoch training loop

if starting_epoch > MAXIMUM_EPOCHS:

    print(
        "\nTraining has already reached the configured maximum "
        f"of {MAXIMUM_EPOCHS} epochs.",
        flush=True,
    )

else:

    print("\n" + "=" * 72, flush=True)
    print("BEGINNING MODEL TRAINING", flush=True)
    print("=" * 72, flush=True)

    print(
        f"\nEpoch range: {starting_epoch}--{MAXIMUM_EPOCHS}",
        flush=True,
    )

    print(
        f"Training batches per epoch: {len(train_loader)}",
        flush=True,
    )

    print(
        f"Validation batches per epoch: {len(validation_loader)}",
        flush=True,
    )

    print(
        "Each epoch displays separate live training and "
        "validation progress bars.",
        flush=True,
    )

    print(
        "The test partition will not be evaluated.",
        flush=True,
    )


    try:

        for epoch in range(
            starting_epoch,
            MAXIMUM_EPOCHS + 1,
        ):

            epoch_start_time = time.time()

            current_learning_rate = float(
                optimizer.param_groups[
                    0
                ]["lr"]
            )


            print("\n" + "=" * 72, flush=True)

            print(
                f"EPOCH {epoch:02d}/{MAXIMUM_EPOCHS}",
                flush=True,
            )

            print("=" * 72, flush=True)

            print(
                "\nPhase 1/4: training batches",
                flush=True,
            )


            # Train on all training participants

            training_result = (
                run_training_epoch_with_progress(
                    model=complete_model,
                    data_loader=train_loader,
                    objective=training_objective,
                    optimizer=optimizer,
                    device=DEVICE,
                    epoch=epoch,
                    maximum_epochs=MAXIMUM_EPOCHS,
                    maximum_gradient_norm=
                        MAXIMUM_GRADIENT_NORM,
                )
            )


            print(
                "\nPhase 2/4: validation batches",
                flush=True,
            )


            # Validate on all validation participants

            validation_result = (
                run_validation_epoch_with_progress(
                    model=complete_model,
                    data_loader=validation_loader,
                    objective=training_objective,
                    device=DEVICE,
                    epoch=epoch,
                    maximum_epochs=MAXIMUM_EPOCHS,
                )
            )


            print(
                "\nPhase 3/4: calculating metrics and "
                "updating the scheduler",
                flush=True,
            )


            # Validation AUC and checkpoint decision

            validation_auc = float(
                validation_result[
                    "metrics"
                ]["final"]["roc_auc"]
            )

            validation_auc_is_valid = bool(
                np.isfinite(
                    validation_auc
                )
            )

            checkpoint_improved = (
                validation_auc_is_valid
                and
                validation_auc
                > best_validation_auc
            )

            if checkpoint_improved:

                best_validation_auc = (
                    validation_auc
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1


            scheduler_score = (
                validation_auc
                if validation_auc_is_valid
                else -1.0
            )

            learning_rate_scheduler.step(
                scheduler_score
            )

            updated_learning_rate = float(
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            epoch_duration_seconds = (
                time.time()
                - epoch_start_time
            )


            # Persistent training history

            history_row = build_training_history_row(
                epoch=epoch,

                learning_rate=
                    current_learning_rate,

                epoch_duration_seconds=
                    epoch_duration_seconds,

                training_result=
                    training_result,

                validation_result=
                    validation_result,

                best_validation_auc=
                    best_validation_auc,

                epochs_without_improvement=
                    epochs_without_improvement,

                checkpoint_improved=
                    checkpoint_improved,
            )

            history_row[
                "learning_rate_after_scheduler"
            ] = updated_learning_rate

            training_history.append(
                history_row
            )

            pd.DataFrame(
                training_history
            ).to_csv(
                TRAINING_HISTORY_PATH,
                index=False,
            )


            print(
                "\nPhase 4/4: saving checkpoints and history",
                flush=True,
            )


            # Save the latest completed epoch

            save_training_checkpoint(
                checkpoint_path=
                    LAST_CHECKPOINT_PATH,

                epoch=epoch,

                model=complete_model,

                optimizer=optimizer,

                scheduler=
                    learning_rate_scheduler,

                best_validation_auc=
                    best_validation_auc,

                epochs_without_improvement=
                    epochs_without_improvement,

                validation_metrics=
                    validation_result[
                        "metrics"
                    ],

                training_history=
                    training_history,
            )


            # Save the best validation checkpoint

            if checkpoint_improved:

                save_training_checkpoint(
                    checkpoint_path=
                        BEST_CHECKPOINT_PATH,

                    epoch=epoch,

                    model=complete_model,

                    optimizer=optimizer,

                    scheduler=
                        learning_rate_scheduler,

                    best_validation_auc=
                        best_validation_auc,

                    epochs_without_improvement=
                        epochs_without_improvement,

                    validation_metrics=
                        validation_result[
                            "metrics"
                        ],

                    training_history=
                        training_history,
                )

                validation_result[
                    "predictions"
                ].to_csv(
                    VALIDATION_PREDICTIONS_PATH,
                    index=False,
                )


            # Readable completed-epoch summary

            train_metrics = training_result[
                "metrics"
            ]["final"]

            validation_metrics = validation_result[
                "metrics"
            ]["final"]


            print("\n" + "-" * 72, flush=True)

            print(
                f"EPOCH {epoch:02d} COMPLETE",
                flush=True,
            )

            print(
                "Duration: "
                f"{epoch_duration_seconds / 60.0:.2f} minutes",
                flush=True,
            )

            print(
                "Learning rate: "
                f"{current_learning_rate:.8f}"
                f" -> {updated_learning_rate:.8f}",
                flush=True,
            )


            print("\nTraining:", flush=True)

            print(
                "  total loss: "
                f"{training_result['losses']['total_loss']:.6f}",
                flush=True,
            )

            print(
                "  ROC AUC: "
                f"{train_metrics['roc_auc']:.6f}",
                flush=True,
            )

            print(
                "  balanced accuracy: "
                f"{train_metrics['balanced_accuracy']:.6f}",
                flush=True,
            )

            print(
                "  mean uncertainty: "
                f"{train_metrics['mean_uncertainty']:.6f}",
                flush=True,
            )

            print(
                "  mean 3MT weight: "
                f"{train_metrics['mean_three_mt_weight']:.6f}",
                flush=True,
            )

            print(
                "  mean effective modalities: "
                f"{training_result['modality_dropout_summary']['mean_effective_modality_count']:.3f}",
                flush=True,
            )

            print(
                "  maximum pre-clipping gradient norm: "
                f"{training_result['gradient_summary']['maximum_gradient_norm_before_clipping']:.6f}",
                flush=True,
            )


            print("\nValidation:", flush=True)

            print(
                "  total loss: "
                f"{validation_result['losses']['total_loss']:.6f}",
                flush=True,
            )

            print(
                "  ROC AUC: "
                f"{validation_metrics['roc_auc']:.6f}",
                flush=True,
            )

            print(
                "  average precision: "
                f"{validation_metrics['average_precision']:.6f}",
                flush=True,
            )

            print(
                "  balanced accuracy: "
                f"{validation_metrics['balanced_accuracy']:.6f}",
                flush=True,
            )

            print(
                "  sensitivity: "
                f"{validation_metrics['sensitivity']:.6f}",
                flush=True,
            )

            print(
                "  specificity: "
                f"{validation_metrics['specificity']:.6f}",
                flush=True,
            )

            print(
                "  Brier score: "
                f"{validation_metrics['brier_score']:.6f}",
                flush=True,
            )

            print(
                "  calibration error: "
                f"{validation_metrics['expected_calibration_error']:.6f}",
                flush=True,
            )

            print(
                "  mean uncertainty: "
                f"{validation_metrics['mean_uncertainty']:.6f}",
                flush=True,
            )

            print(
                "  mean 3MT weight: "
                f"{validation_metrics['mean_three_mt_weight']:.6f}",
                flush=True,
            )


            print("\nCheckpoint status:", flush=True)

            print(
                "  improved this epoch: "
                f"{checkpoint_improved}",
                flush=True,
            )

            print(
                "  best validation ROC AUC: "
                f"{best_validation_auc:.6f}",
                flush=True,
            )

            print(
                "  epochs without improvement: "
                f"{epochs_without_improvement}"
                f"/{EARLY_STOPPING_PATIENCE}",
                flush=True,
            )

            print(
                "  latest completed epoch saved: True",
                flush=True,
            )

            if torch.cuda.is_available():

                print(
                    "  peak CUDA memory: "
                    f"{torch.cuda.max_memory_allocated(0) / (1024 ** 3):.3f} GB",
                    flush=True,
                )


            # Early stopping

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nEarly stopping activated because "
                    "validation ROC AUC did not improve for "
                    f"{EARLY_STOPPING_PATIENCE} consecutive epochs.",
                    flush=True,
                )

                break


    # Interruption and error handling

    except KeyboardInterrupt:

        print(
            "\nTraining was interrupted manually.",
            flush=True,
        )

        print(
            "Only fully completed epochs are recoverable from "
            "the last-epoch checkpoint.",
            flush=True,
        )


    except Exception:

        print(
            "\nTraining stopped because an exception occurred.",
            flush=True,
        )

        if LAST_CHECKPOINT_PATH.exists():
            print(
                "The latest fully completed epoch checkpoint remains saved.",
                flush=True,
            )
        else:
            print(
                "No completed epoch checkpoint exists for this run yet.",
                flush=True,
            )

        traceback.print_exc()

        raise


    # Final training summary

    if len(
        training_history
    ) > 0:

        final_history_table = pd.DataFrame(
            training_history
        )

        completed_epochs = int(
            final_history_table[
                "epoch"
            ].max()
        )

        print("\n" + "=" * 72, flush=True)

        print(
            "TRAINING RUN COMPLETE",
            flush=True,
        )

        print("=" * 72, flush=True)

        print(
            f"\nLast completed epoch: {completed_epochs}",
            flush=True,
        )

        print(
            "Best validation ROC AUC: "
            f"{best_validation_auc:.6f}",
            flush=True,
        )

        print(
            "\nBest checkpoint:\n"
            f"{BEST_CHECKPOINT_PATH}",
            flush=True,
        )

        print(
            "\nLatest checkpoint:\n"
            f"{LAST_CHECKPOINT_PATH}",
            flush=True,
        )

        print(
            "\nTraining history:\n"
            f"{TRAINING_HISTORY_PATH}",
            flush=True,
        )

        print(
            "\nThe test partition has not been evaluated.",
            flush=True,
        )

from datetime import datetime


# Test-output paths

TEST_METRICS_PATH = (
    PREDICTION_DIR
    / "test_metrics.json"
)

TEST_SUMMARY_PATH = (
    PREDICTION_DIR
    / "test_metrics_summary.csv"
)


print("=" * 72)
print("FIXED-EQUAL-FUSION TEST-SET EVALUATION")
print("=" * 72)

print(
    "\nLoading the best validation checkpoint:\n"
    f"{BEST_CHECKPOINT_PATH}"
)


# Load the best validation checkpoint


best_checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

best_checkpoint_epoch = int(
    best_checkpoint[
        "epoch"
    ]
)

best_checkpoint_validation_auc = float(
    best_checkpoint[
        "best_validation_auc"
    ]
)


complete_model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)


print(
    f"\nBest checkpoint epoch: "
    f"{best_checkpoint_epoch}"
)

print(
    "Best validation ROC AUC stored in checkpoint: "
    f"{best_checkpoint_validation_auc:.6f}"
)


# Test evaluation function

def run_test_epoch(
    model,
    data_loader,
    objective,
    device,
    checkpoint_epoch,
):
    """
    Evaluate the frozen model on the untouched test partition.
    """

    model.eval()

    loss_storage = (
        initialise_epoch_loss_storage()
    )

    prediction_storage = defaultdict(
        list
    )

    participant_count = 0


    progress_bar = tqdm(
        enumerate(
            data_loader,
            start=1,
        ),
        total=len(
            data_loader
        ),
        desc="Test evaluation",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )


    with torch.no_grad():

        for batch_number, batch in progress_bar:

            batch = move_nested_batch_to_device(
                batch,
                device,
            )

            targets = batch[
                "target"
            ].long()

            batch_size = int(
                targets.shape[0]
            )

            participant_count += (
                batch_size
            )


            # Frozen forward pass

            model_output = model(
                modalities=batch[
                    "modalities"
                ],

                original_branch_masks=batch[
                    "branch_masks"
                ],
            )


            # Test loss

            loss_output = objective(
                model_output=model_output,
                targets=targets,
                epoch=checkpoint_epoch,
            )


            if not torch.isfinite(
                loss_output[
                    "total_loss"
                ]
            ):

                raise FloatingPointError(
                    "A non-finite test loss was encountered "
                    f"at batch {batch_number}."
                )


            update_epoch_loss_storage(
                storage=loss_storage,
                loss_output=loss_output,
                batch_size=batch_size,
            )


            # Store participant-level outputs

            extracted_predictions = (
                extract_batch_prediction_arrays(
                    batch=batch,
                    model_output=model_output,
                )
            )

            for key, values in (
                extracted_predictions.items()
            ):

                prediction_storage[
                    key
                ].append(
                    values
                )


            running_average_loss = (
                loss_storage[
                    "total_loss"
                ]
                / participant_count
            )

            progress_bar.set_postfix(
                {
                    "avg_loss":
                        f"{running_average_loss:.3f}",

                    "participants":
                        f"{participant_count}/{len(data_loader.dataset)}",
                },
                refresh=True,
            )


    # Finalise test losses and predictions

    mean_losses = finalise_epoch_loss_storage(
        storage=loss_storage,
        participant_count=participant_count,
    )

    concatenated_predictions = (
        concatenate_epoch_prediction_storage(
            prediction_storage
        )
    )

    prediction_metrics = (
        calculate_epoch_prediction_metrics(
            concatenated_predictions
        )
    )

    prediction_table = (
        build_epoch_prediction_table(
            concatenated_predictions=
                concatenated_predictions,

            split_name="test",

            epoch=checkpoint_epoch,
        )
    )


    original_modality_counts = (
        concatenated_predictions[
            "original_modality_count"
        ]
    )

    effective_modality_counts = (
        concatenated_predictions[
            "effective_modality_count"
        ]
    )

    maximum_modality_count_difference = float(
        np.max(
            np.abs(
                original_modality_counts
                - effective_modality_counts
            )
        )
    )


    return {
        "losses":
            mean_losses,

        "metrics":
            prediction_metrics,

        "predictions":
            prediction_table,

        "participant_count":
            int(
                participant_count
            ),

        "batch_count":
            int(
                len(data_loader)
            ),

        "maximum_evaluation_modality_count_difference":
            maximum_modality_count_difference,
    }


# Run the untouched test evaluation

test_result = run_test_epoch(
    model=complete_model,
    data_loader=test_loader,
    objective=training_objective,
    device=DEVICE,
    checkpoint_epoch=best_checkpoint_epoch,
)


# Evaluation-mode modality count

maximum_test_mask_difference = (
    test_result[
        "maximum_evaluation_modality_count_difference"
    ]
)


# Save participant-level test predictions


test_prediction_table = test_result[
    "predictions"
]

test_prediction_table.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False,
)


# Save all test metrics in JSON format

serialisable_test_result = {
    "experiment_name": EXPERIMENT_NAME,
    "project_root": str(PROJECT_ROOT),
    "output_root": str(OUTPUT_ROOT),
    "architecture_preset": ARCHITECTURE_PRESET,
    "random_seed": int(GLOBAL_RANDOM_SEED),
    "availability_gated_cmt": True,
    "task":
        SELECTED_TASK,

    "fold":
        int(
            SELECTED_FOLD
        ),

    "checkpoint_epoch":
        int(
            best_checkpoint_epoch
        ),

    "checkpoint_validation_auc":
        float(
            best_checkpoint_validation_auc
        ),

    "classification_threshold":
        float(
            CLASSIFICATION_THRESHOLD
        ),

    "test_participants":
        int(
            test_result[
                "participant_count"
            ]
        ),

    "test_batches":
        int(
            test_result[
                "batch_count"
            ]
        ),

    "test_losses": {
        key:
            float(value)

        for key, value in (
            test_result[
                "losses"
            ].items()
        )
    },

    "test_metrics":
        test_result[
            "metrics"
        ],

    "maximum_modality_count_difference":
        float(
            maximum_test_mask_difference
        ),

    "evaluated_at":
        datetime.now().isoformat(
            timespec="seconds"
        ),
}


with open(
    TEST_METRICS_PATH,
    "w",
    encoding="utf-8",
) as test_metrics_file:

    json.dump(
        serialisable_test_result,
        test_metrics_file,
        indent=2,
    )


# Build a compact comparison table

test_metric_rows = []

for output_name, display_name in [
    (
        "final",
        "Hybrid",
    ),
    (
        "three_mt",
        "3MT-only",
    ),
    (
        "tmc",
        "TMC-only",
    ),
]:

    output_metrics = test_result[
        "metrics"
    ][
        output_name
    ]

    test_metric_rows.append(
        {
            "OUTPUT":
                display_name,

            "ROC_AUC":
                output_metrics[
                    "roc_auc"
                ],

            "AVERAGE_PRECISION":
                output_metrics[
                    "average_precision"
                ],

            "ACCURACY":
                output_metrics[
                    "accuracy"
                ],

            "BALANCED_ACCURACY":
                output_metrics[
                    "balanced_accuracy"
                ],

            "SENSITIVITY":
                output_metrics[
                    "sensitivity"
                ],

            "SPECIFICITY":
                output_metrics[
                    "specificity"
                ],

            "PRECISION":
                output_metrics[
                    "precision"
                ],

            "F1":
                output_metrics[
                    "f1"
                ],

            "BRIER_SCORE":
                output_metrics[
                    "brier_score"
                ],

            "NEGATIVE_LOG_LIKELIHOOD":
                output_metrics[
                    "negative_log_likelihood"
                ],

            "EXPECTED_CALIBRATION_ERROR":
                output_metrics[
                    "expected_calibration_error"
                ],

            "MEAN_UNCERTAINTY":
                output_metrics[
                    "mean_uncertainty"
                ],
        }
    )


test_metrics_summary = pd.DataFrame(
    test_metric_rows
)

test_metrics_summary.to_csv(
    TEST_SUMMARY_PATH,
    index=False,
)


# Display the final test results

hybrid_test_metrics = test_result[
    "metrics"
][
    "final"
]


print("\n" + "=" * 72)
print(f"FIXED-EQUAL-FUSION FOLD-{SELECTED_FOLD} TEST RESULTS")
print("=" * 72)

print(
    f"\nCheckpoint epoch: "
    f"{best_checkpoint_epoch}"
)

print(
    "Test participants: "
    f"{test_result['participant_count']}"
)

print(
    "Test batches: "
    f"{test_result['batch_count']}"
)

print(
    "Maximum difference between original and effective "
    "test modality counts: "
    f"{maximum_test_mask_difference:.1f}"
)


print(
    "\nFinal hybrid test performance:"
)

print(
    "  ROC AUC: "
    f"{hybrid_test_metrics['roc_auc']:.6f}"
)

print(
    "  average precision: "
    f"{hybrid_test_metrics['average_precision']:.6f}"
)

print(
    "  accuracy: "
    f"{hybrid_test_metrics['accuracy']:.6f}"
)

print(
    "  balanced accuracy: "
    f"{hybrid_test_metrics['balanced_accuracy']:.6f}"
)

print(
    "  sensitivity: "
    f"{hybrid_test_metrics['sensitivity']:.6f}"
)

print(
    "  specificity: "
    f"{hybrid_test_metrics['specificity']:.6f}"
)

print(
    "  precision: "
    f"{hybrid_test_metrics['precision']:.6f}"
)

print(
    "  F1 score: "
    f"{hybrid_test_metrics['f1']:.6f}"
)

print(
    "  Brier score: "
    f"{hybrid_test_metrics['brier_score']:.6f}"
)

print(
    "  negative log-likelihood: "
    f"{hybrid_test_metrics['negative_log_likelihood']:.6f}"
)

print(
    "  expected calibration error: "
    f"{hybrid_test_metrics['expected_calibration_error']:.6f}"
)

print(
    "  mean uncertainty: "
    f"{hybrid_test_metrics['mean_uncertainty']:.6f}"
)

print(
    "  mean 3MT weight: "
    f"{hybrid_test_metrics['mean_three_mt_weight']:.6f}"
)

print(
    "  standard deviation of 3MT weight: "
    f"{hybrid_test_metrics['std_three_mt_weight']:.6f}"
)


print(
    "\nConfusion matrix counts:"
)

print(
    "  true negatives: "
    f"{hybrid_test_metrics['true_negative']}"
)

print(
    "  false positives: "
    f"{hybrid_test_metrics['false_positive']}"
)

print(
    "  false negatives: "
    f"{hybrid_test_metrics['false_negative']}"
)

print(
    "  true positives: "
    f"{hybrid_test_metrics['true_positive']}"
)


print(
    "\nHybrid, 3MT-only, and TMC-only comparison:"
)

display(
    test_metrics_summary.round(
        6
    )
)


print(
    "\nParticipant-level predictions saved to:\n"
    f"{TEST_PREDICTIONS_PATH}"
)

print(
    "\nComplete test metrics saved to:\n"
    f"{TEST_METRICS_PATH}"
)

print(
    "\nCompact metric summary saved to:\n"
    f"{TEST_SUMMARY_PATH}"
)

print(
    "\nThe model was evaluated without gradient updates, "
    "scheduler changes, or test-time modality dropout."
)


# Cleanup
del complete_model, optimizer, training_objective
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nCompleted:")
print(f"  seed={GLOBAL_RANDOM_SEED}")
print(f"  fold={SELECTED_FOLD}")
print(f"  outputs={FOLD_TRAINING_DIR}")
