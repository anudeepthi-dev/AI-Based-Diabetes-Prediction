import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Based Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "diabetes_model_deploy.pkl"
DATASET_PATH = BASE_DIR / "diabetes_012_health_indicators_BRFSS2015.CSV"

FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]

# =========================================================
# 2. LOAD MODEL AND DATASET
# =========================================================

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


model = load_model()

# =========================================================
# 3. APPLICATION HEADER
# =========================================================

st.title("🩺 AI-Based Diabetes Prediction System")

st.markdown(
    """
    This project uses a **Random Forest machine-learning model**
    trained on the CDC BRFSS 2015 Diabetes Health Indicators dataset.

    Enter health-indicator information to obtain a model classification,
    or explore the dataset using the analysis section.
    """
)

st.warning(
    "Educational project only. This application is not a medical "
    "diagnostic tool. Model predictions and probabilities should not "
    "be used to diagnose diabetes or replace professional medical advice."
)

if model is None:
    st.error(
        "Trained model not found. Please run train_model.py first "
        "and verify that model/diabetes_model_deploy.pkl exists."
    )
    st.stop()

# =========================================================
# 4. NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Diabetes Prediction",
        "Dataset Analysis"
    ]
)

# =========================================================
# PAGE 1: DIABETES PREDICTION
# =========================================================

if page == "Diabetes Prediction":

    st.header("Enter Health Information")

    st.write(
        "Complete the form below and click the prediction button."
    )

    with st.form("diabetes_prediction_form"):

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # PERSONAL INFORMATION
        # -------------------------------------------------

        with col1:

            st.subheader("Personal Information")

            age = st.selectbox(
                "Age category",
                options=list(range(1, 14)),
                index=7,
                format_func=lambda x: {
                    1: "18–24",
                    2: "25–29",
                    3: "30–34",
                    4: "35–39",
                    5: "40–44",
                    6: "45–49",
                    7: "50–54",
                    8: "55–59",
                    9: "60–64",
                    10: "65–69",
                    11: "70–74",
                    12: "75–79",
                    13: "80 or older"
                }[x]
            )

            sex = st.selectbox(
                "Sex category",
                options=[0, 1],
                format_func=lambda x: {
                    0: "Female",
                    1: "Male"
                }[x]
            )

            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=12.0,
                max_value=98.0,
                value=27.0,
                step=0.5
            )

            education = st.selectbox(
                "Education level",
                options=list(range(1, 7)),
                format_func=lambda x: {
                    1: "Never attended school or kindergarten only",
                    2: "Grades 1–8",
                    3: "Grades 9–11",
                    4: "High school graduate",
                    5: "Some college or technical school",
                    6: "College graduate"
                }[x]
            )

            income = st.selectbox(
                "Income category",
                options=list(range(1, 9)),
                format_func=lambda x: {
                    1: "Less than $10,000",
                    2: "$10,000–$14,999",
                    3: "$15,000–$19,999",
                    4: "$20,000–$24,999",
                    5: "$25,000–$34,999",
                    6: "$35,000–$49,999",
                    7: "$50,000–$74,999",
                    8: "$75,000 or more"
                }[x]
            )

            general_health = st.selectbox(
                "General health",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: {
                    1: "Excellent",
                    2: "Very good",
                    3: "Good",
                    4: "Fair",
                    5: "Poor"
                }[x]
            )

            mental_health = st.slider(
                "Days of poor mental health in the past 30 days",
                min_value=0,
                max_value=30,
                value=0
            )

            physical_health = st.slider(
                "Days of poor physical health in the past 30 days",
                min_value=0,
                max_value=30,
                value=0
            )

        # -------------------------------------------------
        # HEALTH INDICATORS
        # -------------------------------------------------

        with col2:

            st.subheader("Health Indicators")

            high_bp = st.selectbox(
                "High blood pressure?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            high_chol = st.selectbox(
                "High cholesterol?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            chol_check = st.selectbox(
                "Cholesterol check within the past five years?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            smoker = st.selectbox(
                "Smoked at least 100 cigarettes in your lifetime?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            stroke = st.selectbox(
                "Ever had a stroke?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            heart_disease = st.selectbox(
                "Coronary heart disease or heart attack?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            physical_activity = st.selectbox(
                "Physical activity outside regular work?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            fruits = st.selectbox(
                "Consume fruit at least once per day?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            vegetables = st.selectbox(
                "Consume vegetables at least once per day?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            heavy_alcohol = st.selectbox(
                "Heavy alcohol consumption indicator?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            healthcare = st.selectbox(
                "Have healthcare coverage?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            no_doctor_cost = st.selectbox(
                "Could not see a doctor because of cost?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            difficulty_walking = st.selectbox(
                "Difficulty walking or climbing stairs?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        submitted = st.form_submit_button(
            "🔍 Predict Diabetes Category",
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREPARE INPUT AND PREDICT
    # -----------------------------------------------------

    if submitted:

        input_data = pd.DataFrame(
            [[
                high_bp,
                high_chol,
                chol_check,
                bmi,
                smoker,
                stroke,
                heart_disease,
                physical_activity,
                fruits,
                vegetables,
                heavy_alcohol,
                healthcare,
                no_doctor_cost,
                general_health,
                mental_health,
                physical_health,
                difficulty_walking,
                sex,
                age,
                education,
                income
            ]],
            columns=FEATURES
        )

        try:

            prediction = int(model.predict(input_data)[0])

            probabilities = model.predict_proba(input_data)[0]
            model_classes = list(model.classes_)

            diabetic_index = model_classes.index(1)
            probability = float(probabilities[diabetic_index])

            st.divider()
            st.header("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:
                    st.error(
                        "### Model classification: Diabetic category"
                    )
                else:
                    st.success(
                        "### Model classification: Non-diabetic category"
                    )

            with result_col2:

                st.metric(
                    "Model-estimated probability of the diabetic class",
                    f"{probability:.1%}"
                )

            st.progress(probability)

            st.caption(
                "This is the model's estimated probability for its "
                "diabetic class, not a clinically validated personal "
                "risk percentage or a medical diagnosis."
            )

            # ---------------------------------------------
            # HEALTH SUMMARY
            # ---------------------------------------------

            st.subheader("Health Information Summary")

            summary1, summary2 = st.columns(2)

            with summary1:
                st.write(f"**BMI:** {bmi}")
                st.write(f"**Age category code:** {age}")
                st.write(f"**Sex category code:** {sex}")
                st.write(f"**General health code:** {general_health}")
                st.write(f"**High blood pressure:** {high_bp}")
                st.write(f"**High cholesterol:** {high_chol}")
                st.write(f"**Physical activity:** {physical_activity}")

            with summary2:
                st.write(f"**Education category:** {education}")
                st.write(f"**Income category:** {income}")
                st.write(f"**Poor mental health days:** {mental_health}")
                st.write(f"**Poor physical health days:** {physical_health}")
                st.write(f"**Difficulty walking:** {difficulty_walking}")
                st.write(f"**History of stroke:** {stroke}")
                st.write(f"**Heart disease indicator:** {heart_disease}")

            # ---------------------------------------------
            # CHART OF ENTERED INDICATORS
            # ---------------------------------------------

            st.subheader("Visualization of Entered Health Indicators")

            chart_data = pd.DataFrame({
                "Indicator": [
                    "High blood pressure",
                    "High cholesterol",
                    "Cholesterol check",
                    "Smoker",
                    "Stroke",
                    "Heart disease",
                    "Physical activity",
                    "Fruit consumption",
                    "Vegetable consumption",
                    "Heavy alcohol indicator",
                    "Healthcare coverage",
                    "Doctor cost barrier",
                    "Difficulty walking"
                ],
                "Selected value": [
                    high_bp,
                    high_chol,
                    chol_check,
                    smoker,
                    stroke,
                    heart_disease,
                    physical_activity,
                    fruits,
                    vegetables,
                    heavy_alcohol,
                    healthcare,
                    no_doctor_cost,
                    difficulty_walking
                ]
            })

            st.bar_chart(
                chart_data.set_index("Indicator"),
                height=450
            )

            st.caption(
                "For these binary indicators, 0 means No and 1 means Yes."
            )

        except Exception as error:

            st.error("Unable to generate the prediction.")

            st.write(
                "Check that the trained model uses the expected "
                "21 features in the same order."
            )

            st.exception(error)

# =========================================================
# PAGE 2: DATASET ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    st.header("📊 Diabetes Dataset Analysis")

    if not DATASET_PATH.exists():

        st.error(
            "Dataset file not found. Place "
            "'diabetes_012_health_indicators_BRFSS2015.CSV' "
            "in the same folder as app.py."
        )

        st.stop()

    try:

        dataset = load_dataset(
            str(DATASET_PATH),
            DATASET_PATH.stat().st_mtime
        )

    except Exception as error:

        st.error("Could not read the dataset.")
        st.exception(error)
        st.stop()

    if "Diabetes_012" not in dataset.columns:

        st.error(
            "The dataset does not contain the expected "
            "'Diabetes_012' target column."
        )

        st.stop()

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.subheader("Dataset Overview")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Total records", f"{len(dataset):,}")
    metric2.metric("Total columns", len(dataset.columns))
    metric3.metric(
        "Missing values",
        f"{int(dataset.isnull().sum().sum()):,}"
    )

    st.write("Preview of the dataset:")

    st.dataframe(dataset.head(10), use_container_width=True)

    # -----------------------------------------------------
    # TARGET DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Diabetes Category Distribution")

    target_counts = (
        dataset["Diabetes_012"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )

    target_chart = pd.DataFrame({
        "Category": [
            "No diabetes (0)",
            "Prediabetes (1)",
            "Diabetes (2)"
        ],
        "Number of records": target_counts.values
    })

    st.bar_chart(
        target_chart.set_index("Category"),
        height=350
    )

    st.write("Category counts:")

    st.dataframe(
        target_chart,
        hide_index=True,
        use_container_width=True
    )

    # -----------------------------------------------------
    # BMI DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("BMI Distribution")

    if "BMI" in dataset.columns:

        bmi_data = dataset["BMI"].dropna()

        fig, ax = plt.subplots()

        ax.hist(
            bmi_data,
            bins=30,
            edgecolor="black"
        )

        ax.set_title("Distribution of BMI")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Number of records")

        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------
    # AGE AND DIABETES
    # -----------------------------------------------------

    st.subheader("Diabetes Categories Across Age Groups")

    if "Age" in dataset.columns:

        age_diabetes = pd.crosstab(
            dataset["Age"],
            dataset["Diabetes_012"],
            normalize="index"
        ) * 100

        age_diabetes = age_diabetes.rename(
            columns={
                0: "No diabetes",
                1: "Prediabetes",
                2: "Diabetes"
            }
        )

        st.bar_chart(
            age_diabetes,
            height=400
        )

        st.caption(
            "Bars show the percentage of each diabetes category "
            "within each age category."
        )

    # -----------------------------------------------------
    # BMI AND DIABETES
    # -----------------------------------------------------

    st.subheader("Average BMI by Diabetes Category")

    if "BMI" in dataset.columns:

        average_bmi = (
            dataset.groupby("Diabetes_012")["BMI"]
            .mean()
            .reindex([0, 1, 2])
        )

        average_bmi.index = [
            "No diabetes",
            "Prediabetes",
            "Diabetes"
        ]

        st.bar_chart(
            average_bmi,
            height=350
        )

    # -----------------------------------------------------
    # CORRELATION HEATMAP
    # -----------------------------------------------------

    st.subheader("Correlation Heatmap")

    st.write(
        "This heatmap displays correlations among selected numeric "
        "health indicators and the original diabetes category."
    )

    selected_columns = [
        "Diabetes_012",
        "BMI",
        "Age",
        "HighBP",
        "HighChol",
        "GenHlth",
        "PhysActivity",
        "MentHlth",
        "PhysHlth",
        "DiffWalk"
    ]

    available_columns = [
        column for column in selected_columns
        if column in dataset.columns
    ]

    correlation_data = dataset[available_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(
        correlation_data,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation_data.columns)))
    ax.set_yticks(range(len(correlation_data.columns)))

    ax.set_xticklabels(
        correlation_data.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(correlation_data.columns)

    ax.set_title("Correlation Between Health Indicators")

    fig.colorbar(image, ax=ax, label="Correlation coefficient")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Correlation describes association, not causation. "
        "The original target has three categories: 0, 1, and 2."
    )

    # -----------------------------------------------------
    # MISSING VALUE ANALYSIS
    # -----------------------------------------------------

    st.subheader("Missing Value Analysis")

    missing_data = dataset.isnull().sum()

    missing_data = missing_data[missing_data > 0]

    if missing_data.empty:

        st.success(
            "No missing values were detected in the dataset."
        )

    else:

        missing_frame = missing_data.reset_index()

        missing_frame.columns = [
            "Column",
            "Missing values"
        ]

        st.dataframe(
            missing_frame,
            hide_index=True,
            use_container_width=True
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Diabetes Prediction System | "
    "Random Forest | Streamlit"
)

st.caption(
    "Dataset: CDC BRFSS 2015 Diabetes Health Indicators."
)