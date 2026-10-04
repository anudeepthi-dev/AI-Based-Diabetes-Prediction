import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "diabetes_012_health_indicators_BRFSS2015.CSV"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("AI-BASED DIABETES PREDICTION SYSTEM")
print("=" * 60)

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place the CSV file in the same folder as train_model.py."
    )

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. VALIDATE DATASET
# ============================================================

if "Diabetes_012" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'Diabetes_012' target column."
    )

# This project expects the 21 health-indicator features
# and the original Diabetes_012 target.
expected_feature_count = 21

X = df.drop(columns=["Diabetes_012"]).copy()

if X.shape[1] != expected_feature_count:
    raise ValueError(
        f"Expected {expected_feature_count} input features, "
        f"but found {X.shape[1]}. "
        "Check that you are using the correct dataset."
    )

# Check missing values rather than silently filling them.
missing_total = int(df.isnull().sum().sum())

print("\nTotal missing values:", missing_total)

if missing_total > 0:
    missing_columns = df.columns[df.isnull().any()].tolist()

    raise ValueError(
        "Missing values were found in these columns: "
        f"{missing_columns}. Resolve them before training."
    )

# Check that the target contains only the expected categories.
valid_target_values = {0, 1, 2}
actual_target_values = set(df["Diabetes_012"].unique())

if not actual_target_values.issubset(valid_target_values):
    raise ValueError(
        "Unexpected target values found. "
        f"Expected values from {valid_target_values}, "
        f"but found {actual_target_values}."
    )

if not actual_target_values:
    raise ValueError("The target column is empty.")


# ============================================================
# 4. PREPARE BINARY TARGET
# ============================================================

# Original labels:
# 0 = No diabetes
# 1 = Prediabetes
# 2 = Diabetes

# Binary labels used in this project:
# 0 = No diabetes or prediabetes
# 1 = Diabetes

y = (df["Diabetes_012"] == 2).astype(int)

print("\nBinary target distribution:")
print(y.value_counts().sort_index().rename(
    index={
        0: "Non-Diabetic (0)",
        1: "Diabetic (1)"
    }
))

print("\nNumber of input features before selection:", X.shape[1])


# ============================================================
# 5. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. FEATURE SELECTION
# ============================================================

print("\nSelecting features using Random Forest importance...")

# First, train a model on the training split only.
# This prevents the test split from influencing feature selection.
selector_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

selector_model.fit(X_train, y_train)

# Select features whose importance is at least the median importance.
selector = SelectFromModel(
    selector_model,
    threshold="median",
    prefit=True
)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_feature_mask = selector.get_support()
selected_features = X.columns[selected_feature_mask].tolist()

print("\nSelected features:")
for feature in selected_features:
    print("-", feature)

print("\nNumber of selected features:", len(selected_features))


# ============================================================
# 7. TRAIN FINAL RANDOM FOREST MODEL
# ============================================================

print("\nTraining final Random Forest model...")

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

final_model.fit(X_train_selected, y_train)

print("Model training completed.")


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

y_pred = final_model.predict(X_test_selected)
y_prob = final_model.predict_proba(X_test_selected)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Non-Diabetic", "Diabetic"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# ============================================================
# 9. SAVE FEATURE IMPORTANCE
# ============================================================

print("\nSaving feature importance...")

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

os.makedirs(MODEL_DIR, exist_ok=True)

importance_df.to_csv(IMPORTANCE_PATH, index=False)

print("\nTop 10 selected feature importances:")
print(importance_df.head(10).to_string(index=False))

print(f"\nFeature importance saved to: {IMPORTANCE_PATH}")


# ============================================================
# 10. SAVE SELECTOR AND MODEL TOGETHER
# ============================================================

# IMPORTANT:
# The Streamlit app supplies all 21 original input features.
# Saving the selector and classifier in a single pipeline means
# the app can continue passing the complete input DataFrame.
#
# The selector is already fitted using training data.
# The final classifier was trained on the selected features.

pipeline = Pipeline([
    ("feature_selector", selector),
    ("classifier", final_model)
])

joblib.dump(pipeline, MODEL_PATH)

print(f"\nTrained model saved to: {MODEL_PATH}")

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nYou can now run your Streamlit application.")