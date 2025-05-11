# Getting Started with AI-BRD4-DrugDiscovery

This guide provides step-by-step instructions for setting up and using the AI-BRD4-DrugDiscovery framework.

## Prerequisites

Before using this framework, ensure you have the following prerequisites installed:

- Python 3.8+
- Anaconda or Miniconda (recommended for dependency management)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JJshome/AI-BRD4-DrugDiscovery.git
   cd AI-BRD4-DrugDiscovery
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n brd4-ai python=3.8
   conda activate brd4-ai
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Framework Components

The AI-BRD4-DrugDiscovery framework consists of several components:

### 1. Data Preparation Tools
- `src/data/ligand_selection.py` - Tools for selecting appropriate ligands from ChEMBL database
- `src/data/descriptor_extraction.py` - Utilities for decomposing ligands into molecular descriptors
- `src/data/bioassay_analyzer.py` - Functions for analyzing descriptor distributions in bioassay data

### 2. Data Processing Pipeline
- `src/processing/dimension_reduction.py` - Implementation of dimension reduction techniques (PCA, t-SNE, etc.)
- `src/processing/outlier_detection.py` - Methods for identifying and removing outliers
- `src/processing/feature_selection.py` - Tools for selecting the most informative molecular features

### 3. Machine Learning Models
- `src/models/regression_models.py` - Implementation of ML regression models for predicting IC50 values
- `src/models/evaluation.py` - Functions for evaluating model performance and validation

### 4. Visualization Tools
- `src/visualization/plot_distributions.py` - Tools for visualizing descriptor distributions
- `src/visualization/plot_results.py` - Functions for visualizing prediction results

### 5. Deployment
- `deployment/simulation/` - Interactive simulation environment for testing new candidates

## Basic Usage

### Step 1: Data Collection and Preparation

```python
from src.data.ligand_selection import select_brd4_ligands
from src.data.descriptor_extraction import extract_descriptors
from src.data.bioassay_analyzer import analyze_distributions

# Select BRD4 ligands from ChEMBL database
ligands = select_brd4_ligands(target="BRD4", ic50_max=10000)

# Extract molecular descriptors
descriptors = extract_descriptors(ligands, descriptor_types=["constitutional", "topological", "electronic"])

# Analyze descriptor distributions
distribution_analysis = analyze_distributions(descriptors, bioassay_data=ligands)
```

### Step 2: Data Processing

```python
from src.processing.dimension_reduction import reduce_dimensions
from src.processing.outlier_detection import remove_outliers
from src.processing.feature_selection import select_features

# Reduce dimensionality of descriptor data
reduced_data = reduce_dimensions(descriptors, method="PCA", components=100)

# Remove outliers
cleaned_data = remove_outliers(reduced_data, method="isolation_forest")

# Select most informative features
selected_features = select_features(cleaned_data, method="recursive_feature_elimination", n_features=217)
```

### Step 3: Model Training and Evaluation

```python
from src.models.regression_models import train_model
from src.models.evaluation import evaluate_model

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(selected_features, ligands["IC50"])

# Train machine learning model
model = train_model(X_train, y_train, model_type="gradient_boosting")

# Evaluate model performance
evaluation = evaluate_model(model, X_test, y_test, metrics=["r2", "rmse", "mae"])
print(f"Model R² score: {evaluation['r2']}")
print(f"Model RMSE: {evaluation['rmse']} nM")
```

### Step 4: Prediction and Visualization

```python
from src.models.prediction import predict_activity
from src.visualization.plot_results import plot_prediction_vs_actual

# Predict activity for new compounds
new_compounds = load_compounds("path/to/compounds.sdf")
new_descriptors = extract_descriptors(new_compounds)
processed_descriptors = preprocess_descriptors(new_descriptors)  # Apply same preprocessing as training data
predictions = predict_activity(model, processed_descriptors)

# Visualize results
plot_prediction_vs_actual(predictions, actual_values=None, title="Predicted IC50 Values")
```

## Advanced Usage

For more advanced usage scenarios and detailed examples, refer to the following notebooks in the `examples/` directory:

1. `examples/01_data_exploration.ipynb` - Exploration of BRD4 ligand data and descriptors
2. `examples/02_preprocessing_pipeline.ipynb` - Complete preprocessing pipeline demonstration
3. `examples/03_model_training.ipynb` - Detailed model training and hyperparameter optimization
4. `examples/04_feature_importance.ipynb` - Analysis of important molecular features
5. `examples/05_new_compound_prediction.ipynb` - Predicting activity for new compounds

## Interactive Simulation

To use the interactive simulation environment:

1. Navigate to the deployment directory:
   ```bash
   cd deployment/simulation
   ```

2. Start the simulation server:
   ```bash
   python run_simulation.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

The simulation environment allows you to:
- Upload custom molecular structures
- Modify existing structures using the interactive editor
- Get real-time predictions for BRD4 inhibitory activity
- View detailed analysis of molecular properties

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure all dependencies are correctly installed
2. Verify that input data is in the correct format
3. Check that model files are present in the specified directories

For additional help, please open an issue on the GitHub repository.

## Citation

If you use this framework in your research, please cite:

```
Patent Pending: Artificial Intelligence Method for Developing Novel BRD4 Inhibitors
```

## License

This project is provided for research purposes only. All rights reserved.
