# Baseline Models for Molecular Property Prediction

Reference implementations of classical machine-learning and graph neural network baselines for molecular property prediction. The repository supports classification, regression, and multitask experiments from molecular SMILES.

These are standard baseline architectures. The models do **not** include Holistic GNN mechanisms such as Jumping Knowledge, Virtual Nodes, or additional skip-connection schemes.

## Models

### Classical machine learning

- Random Forest (RF) for classification and regression
- Support Vector Machine (SVM) for classification and regression
- ECFP/Morgan molecular fingerprints generated with RDKit
- Cross-validation and zero-variance feature filtering

### Graph neural networks

- AttentiveFP
- Message Passing Neural Network (MPNN)
- Graph Isomorphism Network (GIN)
- Graph Attention Network (GAT)

## Main features

- Binary and multiclass classification
- Regression
- Mixed and multitask prediction
- Masked losses for missing target values
- Hyperparameter optimization with Optuna
- Model evaluation, calibration, metrics, plots, and embedding analysis
- Random or scaffold-based train/validation/test splits supplied by the user

## Repository structure

```text
components/   Shared neural-network layers and initialization utilities
features/     Molecular fingerprints and molecular graph construction
methods/      Random Forest and SVM pipelines
networks/     AttentiveFP, MPNN, GIN, and GAT implementations
notebooks/    Machine-learning and deep-learning workflows
tests/        Automated tests
train/        Data loading, training, optimization, and prediction
utils/        Metrics, statistics, plots, saving, and analysis utilities
```

The `data/` and `output/` directories are local and are not versioned. Datasets, trained models, embeddings, figures, and optimization results therefore remain outside the Git history.

## Environment installation

Create and activate a Python 3.9 environment:

```bash
conda create --name graph python=3.9
conda activate graph
```

Install PyTorch, PyTorch Geometric, and the remaining dependencies. The command below uses PyTorch 2.5.1 with CUDA 11.8; select a different PyTorch build when required by your system.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

## Input data

Provide CSV files for the training, validation, and test sets. The loaders expect:

1. An identifier in the first column
2. A SMILES string in the second column
3. One or more prediction targets in the remaining columns

Example layout:

```text
data/<dataset>/train.csv
data/<dataset>/val.csv
data/<dataset>/test.csv
```

Split-specific subdirectories, such as `data/<dataset>/scaffold/`, can also be used by updating the notebook paths.

## Usage

Open the appropriate notebook and configure the dataset paths and model settings:

- `notebooks/machine_learning.ipynb` for RF or SVM
- `notebooks/deep_learning.ipynb` for AttentiveFP, MPNN, GIN, or GAT

Run the notebooks from the `notebooks/` directory so that their relative paths resolve correctly.

## Laboratory of Cheminformatics (LCi)

Faculty of Pharmacy  
Federal University of Goiás (UFG)  
Brazil

## Exclusive use

For exclusive use by LCi and its collaborators.

Development team:

- Gustavo Felizardo Santos Sandes — [ORCID: 0000-0002-0591-5133](https://orcid.org/0000-0002-0591-5133)
- Vinícius Alexandre Fiaia Costa — [ORCID: 0000-0001-6479-5963](https://orcid.org/0000-0001-6479-5963)
- Bruno Junior Neves — [ORCID: 0000-0002-1309-8743](https://orcid.org/0000-0002-1309-8743)
