# === train/train.py ===
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- Configuration ---
NUM_EPOCHS = 50
NUM_TRIALS = 50
RANDOM_STATE = 42
MODEL_DIR = "model" # Directory to save model and preprocessor
DATA_PATH = os.path.join('app/data', 'data.csv') # Path to your locally downloaded CSV

# --- Load Dataset ---
try:
    # --- CRITICAL FIX 1: Ensure correct CSV parsing with delimiter and decimal ---
    # The UCI Student Dropout dataset uses semicolons (;) as delimiters.
    # It also does NOT have a header row by default.
    # CRITICAL: It uses COMMA (,) as a decimal separator, not a dot (.).
    df = pd.read_csv(DATA_PATH, sep=';', header=None, decimal=',')

    # The last column (index 36 as it's 0-indexed) is the target.
    # Let's inspect the last column values to be sure.
    # It appears there might be a stray 'Target' label in your actual data.csv file.
    # We need to explicitly handle this to ensure only 'Dropout', 'Enrolled', 'Graduate' are targets.

    # First, let's try to convert all feature columns to numeric.
    # This will turn any non-convertible values into NaN.
    # We will then drop rows with NaNs or fill them.
    X = df.iloc[:, :-1] # All columns except the last one
    y_raw_series = df.iloc[:, -1] # The last column for target

    # Convert all feature columns to numeric, coercing errors to NaN
    # This loop ensures each column is properly converted
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop rows where any of the feature columns became NaN during conversion
    initial_rows = X.shape[0]
    X.dropna(inplace=True)
    y_raw_series = y_raw_series.loc[X.index] # Align y_raw with cleaned X
    if X.shape[0] < initial_rows:
        print(f"Removed {initial_rows - X.shape[0]} rows containing non-numeric data in features.")

    # --- CRITICAL FIX 2: Clean the target variable ---
    # Filter out any non-standard target values like 'Target' string
    valid_targets = ['Dropout', 'Enrolled', 'Graduate']
    y_raw_series = y_raw_series[y_raw_series.isin(valid_targets)]

    # Align X with the cleaned y_raw_series
    X = X.loc[y_raw_series.index]
    y_raw = y_raw_series.values.ravel()

    # If y_raw becomes empty after filtering, there's a serious data issue
    if len(y_raw) == 0:
        raise ValueError("No valid target samples found after cleaning. Check 'data.csv' target column.")


    print(f"Dataset loaded and cleaned successfully from {DATA_PATH}")
    print(f"Final X shape: {X.shape}")
    print(f"Final y_raw shape: {y_raw.shape}")

except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}")
    print("Please ensure 'data.csv' is downloaded and placed in the correct location: 'your_project/app/data/data.csv'.")
    exit()
except Exception as e:
    print(f"An error occurred while loading and cleaning the dataset: {e}")
    print("Please double-check your 'data.csv' file for correct delimiter (';') and decimal separator (',').")
    print("Also ensure the target column only contains 'Dropout', 'Enrolled', or 'Graduate'.")
    exit()

# Store original dtypes for app.py to correctly handle inputs
# After pd.to_numeric, all X columns should now be float/int.
original_X_dtypes_map = X.dtypes.apply(lambda x: str(x)).to_dict()
print("--- Original X dtypes (after numeric conversion) ---")
print(X.dtypes) # This should now show float64 or int64

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# --- Diagnostic: Print Class Distribution ---
print("\n--- Target Class Distribution (y) ---")
class_counts = np.bincount(y)
class_labels = label_encoder.classes_
for i, count in enumerate(class_counts):
    print(f"Class '{class_labels[i]}': {count} samples")
print(f"Total samples: {len(y)}")

# --- CRITICAL FIX 3: Handle single-sample classes for stratified split ---
# We removed "Class 'Target'" with filtering above.
# Now, we only check if any actual outcome class has too few samples for stratification.
single_sample_classes = [idx for idx, count in enumerate(class_counts) if count < 2] # Minimum 2 for split
if single_sample_classes:
    print(f"\nWARNING: The following actual outcome classes have fewer than 2 samples and will cause issues with stratified split:")
    for idx in single_sample_classes:
        print(f"  Class '{class_labels[idx]}'")
    print("Skipping stratified split for train_test_split.")
    use_stratify = False
else:
    use_stratify = True


# Identify column types
# After explicit conversion, all X columns are now numerical for StandardScaler.
categorical_cols_for_onehot = [] # No 'object' dtypes expected after numeric conversion
numerical_cols_for_standardscaler = X.columns.tolist() # All columns are numerical (0, 1, 2, ..., 35)

feature_order = X.columns.tolist() # Preserve the original column order (which are integers)

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_for_standardscaler),
        # 'categorical_cols_for_onehot' will be empty, so no OHE step for raw data strings.
        # If your dataset had actual string columns (e.g., 'city_name'), they would go here.
    ],
    remainder='passthrough' # Ensure other columns are passed through if any (though not expected here)
)
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Fit and transform
X_processed = pipeline.fit_transform(X)

# --- Diagnostic: Print Processed Data Shape and Type ---
print("\n--- Processed Data Info ---")
print(f"Shape of X_processed: {X_processed.shape}")
print(f"Type of X_processed: {type(X_processed)}")
# print("Sample of X_processed (first 5 rows - may be sparse):\n", X_processed.toarray()[:5])

# Get input_dim after preprocessing
input_dim_after_preprocessing = X_processed.shape[1]

# Split
if use_stratify:
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print("\nUsing stratified train/test split.")
else:
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=RANDOM_STATE)
    print("\nSkipping stratified train/test split due to insufficient samples in some classes.")


# Convert to tensors
def to_tensor(data):
    return torch.tensor(data.toarray() if hasattr(data, 'toarray') else data, dtype=torch.float32)

train_X = to_tensor(X_train)
val_X = to_tensor(X_val)
train_y = torch.tensor(y_train, dtype=torch.long)
val_y = torch.tensor(y_val, dtype=torch.long)

# --- Check for CUDA availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Using device: {device} ---")


# Define model (assuming app/model.py contains this exact class definition)
class DropoutPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
best_preproc_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Optuna objective
def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])

    model = DropoutPredictorNet(input_dim_after_preprocessing, hidden_dim, dropout_rate, len(np.unique(y))).to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Use weighted CE Loss if uncommented above
    # class_weights_tensor is calculated previously based on final `y`
    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    criterion = nn.CrossEntropyLoss()

    # Move data to device
    train_X_dev = train_X.to(device)
    train_y_dev = train_y.to(device)
    val_X_dev = val_X.to(device)
    val_y_dev = val_y.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        logits = model(train_X_dev)
        loss = criterion(logits, train_y_dev)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_X_dev)
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = accuracy_score(val_y.numpy(), val_preds.cpu().numpy())
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    model.eval()
    with torch.no_grad():
        val_logits = model(val_X_dev)
        val_preds = torch.argmax(val_logits, dim=1)
        acc = accuracy_score(val_y.numpy(), val_preds.cpu().numpy())

    # Save best model & pipeline along with essential metadata
    if acc > objective.best_acc:
        print(f"\n--- New best accuracy found: {acc:.4f} (Trial {trial.number}) ---")
        torch.save(model.state_dict(), best_model_path)
        joblib.dump(
            {
                'pipeline': pipeline,
                'label_encoder': label_encoder,
                'feature_order': feature_order,
                'numerical_cols': numerical_cols_for_standardscaler,
                'categorical_cols': categorical_cols_for_onehot, # This will be empty list now
                'input_dim': input_dim_after_preprocessing,
                'best_params': trial.params,
                'original_X_dtypes_map': original_X_dtypes_map
            },
            best_preproc_path
        )
        objective.best_acc = acc

    return acc

objective.best_acc = 0.0

# Run optimization
print(f"\n--- Starting Optuna Optimization for {NUM_TRIALS} trials ---")
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
study.optimize(objective, n_trials=NUM_TRIALS)

# Print results
print("\n✅ Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")
print(f"\n📈 Best Validation Accuracy: {study.best_value:.4f}")
print(f"Best Trial Number: {study.best_trial.number}")
print("\n--- Training Finished ---")