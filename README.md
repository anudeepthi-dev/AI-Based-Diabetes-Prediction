# AI-Based Diabetes Prediction System

## 1. Project Overview

The **AI-Based Diabetes Prediction System** is an educational machine-learning project that uses a Random Forest classifier to classify health-indicator information into two categories:

- **Class 0 — Non-diabetic category:** Combines the original dataset's `No diabetes (0)` and `Prediabetes (1)` labels.
- **Class 1 — Diabetic category:** Corresponds to the original dataset's `Diabetes (2)` label.

The Streamlit web application has two pages:

1. **Diabetes Prediction** — accepts health-indicator inputs and displays the model's predicted category, its estimated probability for the diabetic class, a summary of the inputs, and a chart of selected binary indicators.
2. **Dataset Analysis** — displays dataset information and visualizations, including target-category counts, BMI distribution, diabetes categories across age groups, average BMI by category, a correlation heatmap, and missing-value analysis.

> **Important:** This is an educational demonstration, not a medical diagnostic tool. Predictions and model probabilities are not clinically validated measures of an individual's diabetes risk. Do not use them for medical decisions.

## 2. Project Objectives

- Build a supervised machine-learning classifier using health-indicator data.
- Train and evaluate a Random Forest model.
- Provide an interactive web interface for entering health indicators.
- Display a predicted category and the model-estimated probability of the diabetic class.
- Explore the dataset with charts and correlation analysis.
- Document the dataset, model evaluation, setup, and limitations.

## 3. Dataset

**Expected file name:** `diabetes_012_health_indicators_BRFSS2015.CSV`

The dataset used is the CDC BRFSS 2015 Diabetes Health Indicators dataset. The local file used during development contains:

- **253,680 records**
- **22 columns:** 21 input features and one target column, `Diabetes_012`
- **0 missing values** detected in the dataset used during development

The original target column has three values:

| Original value | Meaning | Count |
|---:|---|---:|
| 0 | No diabetes | 213,703 |
| 1 | Prediabetes | 4,631 |
| 2 | Diabetes | 35,346 |
| **Total** | | **253,680** |

Dataset source: [UCI Machine Learning Repository — CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).

### Input features

The model expects these 21 columns in this order:

`HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`.

These are survey-based health indicators and demographic categories, not current clinical laboratory measurements. The dataset does not include a glucose measurement column, so the application does not ask for glucose.

## 4. Target Label Preparation

The training script converts the original three-category target into a binary target:

- Original `Diabetes_012 = 0` → model class `0`
- Original `Diabetes_012 = 1` → model class `0`
- Original `Diabetes_012 = 2` → model class `1`

Therefore, **the model's non-diabetic class includes the original prediabetes category**. This simplification should be considered when interpreting results.

## 5. Machine-Learning Method and Preprocessing

The project uses a **Random Forest Classifier** from scikit-learn. The current training workflow includes:

1. Loading the CSV dataset using Pandas.
2. Validating that the target column exists and that the expected 21 input features are present.
3. Checking the target values and checking for missing values. The script stops with an error if missing values are found; it does not impute them.
4. Converting the original three-category target into the binary target described above.
5. Splitting data into training and test sets with an 80/20 split and stratification, using `random_state=42`.
6. Training a Random Forest model on the training set to estimate feature importance.
7. Selecting features with `SelectFromModel` using the median feature-importance threshold.
8. Training a final Random Forest classifier using the selected features.
9. Evaluating the final model on the held-out test set.
10. Saving the feature-selection and classification steps together as a pipeline in `model/diabetes_model.pkl`.
11. Saving the selected features' importance values to `model/feature_importance.csv`.

The Random Forest settings used for the selector and final classifier are `n_estimators=100`, `random_state=42`, `class_weight="balanced"`, and `n_jobs=-1`.

**Other preprocessing:** Numeric scaling and categorical encoding are not performed in this workflow. Random Forest generally does not require feature scaling, but that does not mean the input features have been normalized or standardized.

## 6. Model Evaluation

The Random Forest classification algorithm was used to train and evaluate the diabetes prediction model. Feature selection was performed using Random Forest feature importance to identify the most relevant input features.

The dataset was divided into **80% training data and 20% testing data**. The model was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

### 6.1 Model Performance

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 80.12% |
| Precision | 33.66% |
| Recall | 43.95% |
| F1-score | 38.12% |
| ROC-AUC | 76.97% |

### 6.2 Selected Features

The feature selection process identified the following 11 features for training the final model:

1. `HighBP`
2. `HighChol`
3. `BMI`
4. `Smoker`
5. `Fruits`
6. `GenHlth`
7. `MentHlth`
8. `PhysHlth`
9. `Age`
10. `Education`
11. `Income`

### 6.3 Classification Report

The model's classification results on the test dataset were:

| Class | Precision | Recall | F1-score | Test Records |
|---|---:|---:|---:|---:|
| Non-Diabetic | 0.90 | 0.86 | 0.88 | 43,667 |
| Diabetic | 0.34 | 0.44 | 0.38 | 7,069 |

### 6.4 Confusion Matrix

The confusion matrix produced the following results:

| Actual Class | Predicted Non-Diabetic | Predicted Diabetic |
|---|---:|---:|
| Non-Diabetic | 37,543 | 6,124 |
| Diabetic | 3,962 | 3,107 |

The model correctly classified 37,543 non-diabetic cases and 3,107 diabetic cases in the test dataset.

### 6.5 Feature Importance

The top features contributing to the model's predictions were:

| Feature | Importance |
|---|---:|
| BMI | 0.2143 |
| Age | 0.1514 |
| GenHlth | 0.1136 |
| Income | 0.1045 |
| PhysHlth | 0.0921 |
| HighBP | 0.0835 |
| Education | 0.0711 |
| MentHlth | 0.0699 |
| HighChol | 0.0399 |
| Fruits | 0.0300 |

BMI had the highest feature importance among the selected features, followed by Age and GenHlth.

### 6.6 Evaluation Summary

The model achieved **80.12% accuracy** and a **76.97% ROC-AUC score** on the test dataset. However, its recall for the diabetic class was 43.95%, meaning it identified fewer than half of the diabetic cases in the test set. Its precision for that class was 33.66%.

These results indicate that the model has limitations in identifying diabetic cases. It is suitable as an educational machine-learning project, but its predictions should not be used for medical diagnosis or treatment decisions.

## 7. Application Features

### Page 1: Diabetes Prediction

The Streamlit form accepts the 21 features used by the trained model, including:

- Age category, sex category, BMI, education, and income
- High blood pressure and high cholesterol indicators
- Cholesterol-check indicator
- Smoking, stroke, and heart-disease indicators
- Physical activity, fruit and vegetable consumption, and heavy-alcohol indicator
- Healthcare coverage and cost-related barrier to seeing a doctor
- General health, poor mental-health days, poor physical-health days, and difficulty walking

After submission, the app displays:

- The model's binary category prediction
- The model-estimated probability for the diabetic class
- A summary of selected health information
- A bar chart of selected binary health indicators

The displayed probability is the classifier's estimate and is not necessarily calibrated. It is **not** a validated personal medical-risk percentage.

### Page 2: Dataset Analysis

The analysis page displays:

- Dataset size, number of columns, and total missing values
- A preview of the dataset
- Counts for all three original target categories
- A BMI histogram
- Diabetes-category percentages across age categories
- Average BMI by original diabetes category
- A correlation heatmap for selected numeric health indicators
- Missing-value counts for columns with missing data, if any

Correlation indicates association and does not establish causation.

## 8. Project Structure

```text
AI-Based-Diabetes-Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── diabetes_012_health_indicators_BRFSS2015.CSV
└── model/
    ├── diabetes_model.pkl
    └── feature_importance.csv
```

Keep the dataset filename consistent with the path configured in `train_model.py` and the path used by `app.py`.

## 9. Requirements

The `requirements.txt` file should contain:

```text
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
```

## 10. Installation and Running

### Step 1: Open the project folder

Open the `AI-Based-Diabetes-Prediction` directory in VS Code.

### Step 2: (Recommended) Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Train the model (only if the saved model is missing or you want to retrain)

```bash
python train_model.py
```

After successful training, the script saves:

- `model/diabetes_model.pkl`
- `model/feature_importance.csv`

### Step 5: Run the web application

```bash
python -m streamlit run app.py
```

Streamlit will display a local address, usually `http://localhost:8501`. Open that address in your browser. Keep the terminal process running while using the application.

## 11. Limitations

- The project uses survey-derived indicators from the BRFSS 2015 dataset, not current clinical examination or laboratory data.
- There is no glucose feature in this dataset.
- Prediabetes is grouped into the model's non-diabetic class for binary classification.
- The dataset has an imbalanced target distribution.
- The model may miss diabetic-class cases.
- The displayed model probability is not clinically calibrated or validated as an individual risk estimate.
- Correlation charts show association, not causation.
- The model is for educational use only and must not be used to make medical decisions.

## 12. Conclusion

This project demonstrates an educational machine-learning workflow: validating and preparing a binary target, selecting features, training and evaluating a Random Forest classifier, saving the model pipeline, building an interactive Streamlit interface, and exploring health-indicator patterns through visualizations.

Further work could investigate probability calibration, alternative approaches to class imbalance, and more thorough external evaluation. Any changes should be tested and documented before being described as completed.
