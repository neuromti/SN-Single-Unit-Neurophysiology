# Analysis scripts for: Spatially resolved single-neuron physiology of the human substantia nigra in Parkinson’s disease
Python scripts for microelectrode recording (MER) analyses of substantia nigra neurophysiology in Parkinson's disease during deep brain stimulation surgery.

## Overview
This repository contains the core pipeline used to preprocess, filter, and analyze intraoperative electrophysiological data. This repository contains analysis scripts without patient-level data.
Core analysis code written by **Tameem M. Alozzi**. Valuable assistance and contributions were provided by co-authors **Farzin Negahbani** and **Enrico Ferrea**.

## Repository Layout

```text
.
├── SN_Neurophys_Part_1_preprocess_analysis.py               # Preprocessing and electrophysiological, spatial, waveform morphology analyses, and statistics
├── SN_Neurophys_Part_2_preoperative_clinical_corr_stats.py  # MDS-UPDRS-III Clinical correlation and statistics
├── requirements.txt                                         # Python dependencies
└── data/                                                    # Local/private data, ignored by git
    ├── spikes_neurons/                                      # Spike-sorted output folders (tridesclous DataIO format)
    │   └── <patient_id>/
    │       └── <recording>/
    ├── LFPs/                                                # Raw LFP recordings in Spike2 .smr format
    │   └── <patient_id>/
    │       └── <recording>.smr
    └── MER_features_coords_mapper_with_names_distances.csv  #MNI coordinates of recording locations and respective atlas-based annotations
```

## Data Placement

The code expects data to be placed locally using the structure above. At minimum before running Part 1; 

-  The coordinate/atlas mapping file should be placed in `data/` : `MER_features_coords_mapper_with_names_distances.csv`

-  Raw spike-sorted recordings should be placed in `data/spikes_neurons/`

-  Raw LFP files in `data/LFPs/`

Part 2 additionally requires a preoperative clinical scores file (MDS-UPDRS III) and a merged cell feature table produced by Part 1. Local paths in both scripts should be updated to match your directory structure before running.

## Running

Install the Python dependencies listed in requirements.txt, then run the scripts in order:
-  python **SN_Neurophys_Part_1_preprocess_analysis.py**
-  python **SN_Neurophys_Part_2_preoperative_clinical_corr_stats.py**

Both scripts are organized into clearly labelled **#%%** sections and can also be run interactively cell-by-cell in an IDE such as Spyder or VS Code.









