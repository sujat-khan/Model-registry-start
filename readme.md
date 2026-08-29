# MLflow Model Registry - Hands-on Guide & Revision Notes

A practical step-by-step walkthrough of managing the complete Machine Learning lifecycle using **MLflow Tracking** and **MLflow Model Registry**.

---

## Why MLflow Model Registry?

When training multiple models across experiments, tracking metrics alone is not enough. In production, you need:
1. **Centralized Model Hub**: Store, version, and manage models in one place.
2. **Model Versioning**: Keep track of every iteration (`v1`, `v2`, `v3`...) automatically.
3. **Stage Transitions**: Promote models through lifecycle stages (`None` -> `Staging` -> `Production` -> `Archived`).
4. **Metadata & Governance**: Tag models, add descriptions, and track who trained what and with which data.
5. **Seamless Serving / Inference**: Load the latest production model without hardcoding file paths.

---

## Project Structure

```
model-registry-start/
├── train.py              # Hyperparameter tuning, experiment tracking & model logging
├── register_model.py     # Programmatic model registration, descriptions & tags
├── stage_transition.py   # Promoting model versions between stages (Staging / Production)
├── inference.py          # Loading registered models and running predictions
├── requirements.txt      # Project dependencies
├── .gitignore            # Ignored files (virtualenv, artifacts, DB, PDFs)
└── readme.md             # Project documentation & revision notes
```

---

## Step-by-Step Workflow

### Step 1: Environment Setup & Starting MLflow Server

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the MLflow UI & Backend Store:**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```
   > Open `http://127.0.0.1:5000` in your browser to access the dashboard.

---

### Step 2: Training & Tracking (`train.py`)

Trains a `RandomForestClassifier` on the **Pima Indians Diabetes Dataset** using `GridSearchCV`.

**Key Concepts Applied:**
- **Nested Runs**: A parent run for the overall experiment and child runs for each hyperparameter combination (`n_estimators`, `max_depth`).
- **Dataset Logging**: Logs training and testing datasets via `mlflow.data.from_pandas()` for data lineage.
- **Model Signatures**: Infers input schema and output tensors (`infer_signature(X_train, predictions)`) to enforce data consistency during inference.
- **Artifacts & Tags**: Logs source code (`train.py`) and author metadata.

**Run training:**
```bash
python train.py
```

---

### Step 3: Registering the Model (`register_model.py`)

Takes the best model from a specific training run and registers it into the MLflow Model Registry under the name `diabetes-rf`.

**Key Concepts Applied:**
- **Model URI Syntax**: `runs:/{run_id}/{artifact_path}` (e.g., `runs:/ef32f.../random_forest`).
- **Adding Metadata**:
  - `client.update_model_version()`: Adds a clear description of the model version.
  - `client.set_model_version_tag()`: Adds custom tags (e.g., `experiment: diabetes prediction`, `day: sat`).

**Run registration:**
```bash
python register_model.py
```

---

### Step 4: Model Stage Transitions (`stage_transition.py`)

Manages the lifecycle of model versions using `MlflowClient`.

**Available Stages:**
- `None` (Default upon registration)
- `Staging` (Ready for testing/validation)
- `Production` (Serving live predictions)
- `Archived` (Deprecated/superseded versions)

**Code Highlight:**
```python
client.transition_model_version_stage(
    name="diabetes-rf",
    version=3,
    stage="Production",
    archive_existing_versions=True  # Automatically demotes current Production model to Archived
)
```

**Run transition:**
```bash
python stage_transition.py
```

---

### Step 5: Loading & Inference (`inference.py`)

Loads the model directly from the Model Registry without needing local `.pkl` files.

**How to load models:**
- By version: `models:/diabetes-rf/3`
- By stage: `models:/diabetes-rf/Production`

```python
import mlflow.pyfunc
import pandas as pd

# Load model from registry
model = mlflow.pyfunc.load_model(model_uri="models:/diabetes-rf/3")

# Input data must match the logged signature columns
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
sample_data = pd.DataFrame([[1, 85, 66, 29, 0, 26.6, 0.351, 31]], columns=columns)

# Predict
predictions = model.predict(sample_data)
print("Prediction:", predictions)
```

**Run inference:**
```bash
python inference.py
```

---

## Important Gotchas & Revision Summary

1. **Why `model.skops` instead of `model.pkl` in MLflow 3.x?**
   - MLflow 3.x uses `skops` serialization by default for scikit-learn models for security reasons.
   - If you specifically need `.pkl`, pass `serialization_format="cloudpickle"` in `mlflow.sklearn.log_model()`.

2. **Why pass a `pandas.DataFrame` during inference?**
   - When you log a model with a signature inferred from a DataFrame, MLflow expects named features. Passing a raw NumPy array will trigger `SCHEMA_ENFORCEMENT_FAILED`.

3. **`runs:/` vs `models:/` URI:**
   - Use `runs:/<run_id>/<artifact_path>` to register or load an artifact from a specific experiment run.
   - Use `models:/<model_name>/<version_or_stage>` to load an approved model from the registry.

---

## Reference
- [CampusX MLflow Model Registry Repository](https://github.com/campusx-official/mlflow-model-registry-demo/tree/master)
