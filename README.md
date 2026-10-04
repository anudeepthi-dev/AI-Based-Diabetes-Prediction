# AI-Based Diabetes Prediction System

## 1. Project Overview

The **AI-Based Diabetes Prediction System** is an educational machine-learning project that uses a **Random Forest Classifier** to classify health-indicator information into two categories:

- **Class 0 — Non-Diabetic:** Combines the original dataset's No Diabetes (0) and Prediabetes (1) categories.
- **Class 1 — Diabetic:** Corresponds to the original dataset's Diabetes (2) category.

The project provides an interactive **Streamlit web application** with two main pages:

### Diabetes Prediction
Accepts health-indicator inputs and displays:

- Predicted diabetes category
- Model-estimated probability for the diabetic class
- Summary of selected inputs
- Visualization of selected binary health indicators

### Dataset Analysis
Provides exploratory analysis and visualizations, including:

- Dataset information
- Diabetes-category distribution
- BMI distribution
- Diabetes categories across age groups
- Average BMI by diabetes category
- Correlation heatmap
- Missing-value analysis

> **Important:** This project is intended for educational and demonstration purposes only. It is **not a medical diagnostic tool**. Model predictions and probabilities are not clinically validated and must not be used for medical diagnosis, treatment, or healthcare decisions.

---

## 2. Project Objectives

The main objectives of this project are:

- Build a supervised machine-learning classifier using health-indicator data.
- Train and evaluate a Random Forest model.
- Perform feature selection using Random Forest feature importance.
- Provide an interactive web interface for entering health indicators.
- Display a predicted diabetes category and model-estimated probability.
- Explore the dataset using charts and correlation analysis.
- Demonstrate an end-to-end machine-learning workflow from data preparation to deployment.
- Document the model evaluation, application features, and limitations.

---

## 3. Dataset

The project uses the **CDC BRFSS 2015 Diabetes Health Indicators dataset**.

### Dataset File

The dataset included in the project is:

```text
diabetes_012_health_indicators_BRFSS2015.csv
```

### Dataset Information

- **Records:** 253,680
- **Columns:** 22
- **Input features:** 21
- **Target column:** `Diabetes_012`
- **Missing values:** 0 detected in the dataset used during development

### Original Target Categories

| Original Value | Meaning | Count |
|---|---|---:|
| 0 | No Diabetes | 213,703 |
| 1 | Prediabetes | 4,631 |
| 2 | Diabetes | 35,346 |
| **Total** | | **253,680** |

The dataset contains survey-based health indicators and demographic categories. It does **not** contain a glucose measurement column, so the application does not request glucose as an input.

### Input Features

The model expects the following 21 input features:

```text
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
```

---

## 4. Target Label Preparation

The original dataset contains three target categories. For this project, the target was converted into a binary classification problem.

| Original `Diabetes_012` | Model Class | Meaning |
|---|---|---|
| 0 | 0 | Non-Diabetic |
| 1 | 0 | Non-Diabetic |
| 2 | 1 | Diabetic |

Therefore:

```text
Original 0 → Model 0
Original 1 → Model 0
Original 2 → Model 1
```

The model's **Non-Diabetic** class therefore includes both the original No Diabetes and Prediabetes categories.

This simplification should be considered when interpreting the model's results.

---

## 5. Machine-Learning Method

The project uses a **Random Forest Classifier** from `scikit-learn`.

### Training Workflow

The training process includes:

1. Loading the dataset using Pandas.
2. Validating the target column.
3. Validating the expected input features.
4. Checking target values.
5. Checking for missing values.
6. Converting the original three-category target into a binary target.
7. Splitting the dataset into training and testing sets.
8. Performing feature selection using Random Forest feature importance.
9. Training the final Random Forest classifier using the selected features.
10. Evaluating the model on the test dataset.
11. Saving the trained model pipeline.
12. Saving feature-importance information.

### Data Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

The split uses stratification and:

```text
random_state = 42
```

### Random Forest Configuration

The original training configuration uses:

```text
n_estimators = 100
random_state = 42
class_weight = "balanced"
n_jobs = -1
```

Feature selection is performed using `SelectFromModel` with the median feature-importance threshold.

> **Deployment note:** A smaller deployment model, `diabetes_model_deploy.pkl`, is included in the repository so that the application can be hosted within GitHub's file-size limits. The original larger model remains available locally for development.

Random Forest does not require numerical feature scaling for this workflow, so normalization or standardization is not performed.

---

## 6. Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 6.1 Model Performance

| Evaluation Metric | Result |
|---|---:|
| Accuracy | **80.12%** |
| Precision | **33.66%** |
| Recall | **43.95%** |
| F1-score | **38.12%** |
| ROC-AUC | **76.97%** |

### 6.2 Selected Features

The feature-selection process identified the following important features:

1. HighBP
2. HighChol
3. BMI
4. Smoker
5. Fruits
6. GenHlth
7. MentHlth
8. PhysHlth
9. Age
10. Education
11. Income

### 6.3 Classification Report

| Class | Precision | Recall | F1-score | Test Records |
|---|---:|---:|---:|---:|
| Non-Diabetic | 0.90 | 0.86 | 0.88 | 43,667 |
| Diabetic | 0.34 | 0.44 | 0.38 | 7,069 |

### 6.4 Confusion Matrix

| Actual Class | Predicted Non-Diabetic | Predicted Diabetic |
|---|---:|---:|
| Non-Diabetic | 37,543 | 6,124 |
| Diabetic | 3,962 | 3,107 |

The model correctly classified:

- **37,543** non-diabetic cases
- **3,107** diabetic cases

### 6.5 Feature Importance

The major features contributing to the model's predictions included:

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

**BMI** had the highest feature importance among the selected features, followed by **Age** and **General Health (`GenHlth`)**.

### 6.6 Evaluation Summary

The model achieved an accuracy of **80.12%** and an ROC-AUC of **76.97%** on the test dataset.

However, the recall for the diabetic class was **43.95%**, meaning the model did not identify all diabetic cases correctly. The precision for the diabetic class was **33.66%**.

Therefore, the model is suitable as an **educational machine-learning demonstration**, but its predictions should not be interpreted as medical diagnoses or validated individual health-risk estimates.

---

## 7. Application Features

The application is developed using **Streamlit**.

### Page 1 — Diabetes Prediction

The application accepts the 21 features used by the trained model, including:

- Age category
- Sex
- BMI
- Education
- Income
- High blood pressure
- High cholesterol
- Cholesterol check
- Smoking
- Stroke history
- Heart disease or heart attack
- Physical activity
- Fruit consumption
- Vegetable consumption
- Heavy alcohol consumption
- Healthcare coverage
- Cost-related barrier to seeing a doctor
- General health
- Poor mental-health days
- Poor physical-health days
- Difficulty walking

After submitting the form, the application displays:

- Predicted category
- Estimated probability for the diabetic class
- Summary of selected health information
- Bar chart of selected binary health indicators

The displayed probability is the classifier's estimated probability and is **not a clinically calibrated individual diabetes-risk percentage**.

### Page 2 — Dataset Analysis

The Dataset Analysis page displays:

- Dataset size
- Number of columns
- Total missing values
- Dataset preview
- Original diabetes-category distribution
- BMI histogram
- Diabetes-category percentages across age groups
- Average BMI by diabetes category
- Correlation heatmap
- Missing-value analysis

Correlation represents association between variables and does not establish causation.

---

## 8. Project Structure

The current project structure is:

```text
AI-Based-Diabetes-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── diabetes_012_health_indicators_BRFSS2015.csv
│
└── model/
    ├── diabetes_model_deploy.pkl
    └── feature_importance.csv
```

### Local Development Model

The original larger model file may also exist locally:

```text
model/diabetes_model.pkl
```

This large file is intentionally excluded from GitHub because GitHub has a hard file-size limit.

The deployment version is:

```text
model/diabetes_model_deploy.pkl
```

---

## 9. Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Matplotlib**
- **Git**
- **GitHub**

---

## 10. Requirements

The `requirements.txt` file contains the main dependencies required to run the project:

```text
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
```

---

## 11. Installation and Running Locally

### Step 1 — Open the Project

Open the following folder in VS Code:

```text
AI-Based-Diabetes-Prediction
```

### Step 2 — Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4 — Train the Model

Training is only required if the saved model is missing or the model needs to be retrained.

```powershell
python train_model.py
```

The training script generates the model and feature-importance information.

### Step 5 — Run the Streamlit Application

```powershell
python -m streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

Keep the terminal running while using the application.

---

## 12. Deployment

The project is deployed using **Streamlit Community Cloud**.

The application source code and deployment model are maintained in the GitHub repository.

### Deployment Components

- GitHub repository
- Streamlit application
- `app.py`
- `requirements.txt`
- Dataset CSV
- Deployment model
- Feature-importance CSV

The application uses relative paths so that the dataset and model can be accessed correctly when deployed.

The dataset filename must remain exactly:

```text
diabetes_012_health_indicators_BRFSS2015.csv
```

File-name capitalization should not be changed because deployment environments such as Linux are case-sensitive.

---

## 13. Limitations

This project has several limitations:

1. The dataset contains survey-derived health indicators rather than current clinical examination or laboratory measurements.
2. The dataset does not contain a glucose measurement.
3. Prediabetes is grouped into the Non-Diabetic class for binary classification.
4. The target classes are imbalanced.
5. The model may incorrectly classify some diabetic cases as non-diabetic.
6. The model's probability output is not clinically calibrated.
7. The dataset represents BRFSS 2015 data and may not represent current populations or healthcare conditions.
8. Correlation analysis shows association and does not establish causation.
9. The model has not been clinically validated.
10. The system should not be used to make medical or treatment decisions.

---

## 14. Future Enhancements

Possible future improvements include:

- Probability calibration.
- Advanced techniques for handling class imbalance.
- Hyperparameter optimization.
- Testing additional machine-learning algorithms.
- External validation using newer datasets.
- Improved model interpretability.
- Additional visualization and analytics.
- More comprehensive feature engineering.
- Separate classification of No Diabetes, Prediabetes, and Diabetes.
- Integration with clinically validated datasets where appropriate.

Any future enhancement should be properly tested and evaluated before being presented as a completed feature.

---

## 15. Conclusion

The **AI-Based Diabetes Prediction System** demonstrates an end-to-end machine-learning workflow using health-indicator data.

The project includes:

- Dataset validation and preparation
- Binary target transformation
- Feature selection
- Random Forest classification
- Model evaluation
- Feature-importance analysis
- Interactive Streamlit interface
- Dataset visualization
- GitHub-based source-code management
- Streamlit Cloud deployment

The project demonstrates how machine-learning techniques can be applied to survey-based health data while also highlighting the importance of understanding model limitations, class imbalance, and responsible interpretation of predictions.

> **Disclaimer:** This application is developed for educational purposes only. It is not a medical diagnostic system, and its predictions must not be used for medical diagnosis, treatment, or healthcare decisions.