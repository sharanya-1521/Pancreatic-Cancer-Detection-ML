
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Pancreatic Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pancreatic Cancer Detection")
st.write(
    "This educational application demonstrates two machine learning models: "
    "cancer detection and cancer stage prediction."
)

st.warning(
    "Educational project only. These models are not validated for medical "
    "diagnosis or treatment decisions. The two models use different datasets."
)

@st.cache_resource
def load_detection_model():
    return joblib.load("pancreatic_detection_model.joblib")

@st.cache_resource
def load_stage_model():
    return joblib.load("pancreatic_stage_model.joblib")

@st.cache_data
def load_stage_dataset():
    return pd.read_csv("Pancreatic_Cancer.csv")

try:
    detection_model = load_detection_model()
    stage_model = load_stage_model()
    stage_df = load_stage_dataset()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.stop()

# -----------------------------------
# STEP 1: CANCER DETECTION
# -----------------------------------

st.header("Step 1: Cancer Detection")

st.write("Enter the patient and biomarker details.")

with st.form("detection_form"):
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=60
    )

    sex = st.selectbox("Sex", ["M", "F"])

    plasma_CA19_9 = st.number_input(
        "Plasma CA19-9", min_value=0.0, value=0.0
    )

    creatinine = st.number_input(
        "Creatinine", min_value=0.0, value=0.0
    )

    LYVE1 = st.number_input(
        "LYVE1", min_value=0.0, value=0.0
    )

    REG1B = st.number_input(
        "REG1B", min_value=0.0, value=0.0
    )

    TFF1 = st.number_input(
        "TFF1", min_value=0.0, value=0.0
    )

    REG1A = st.number_input(
        "REG1A", min_value=0.0, value=0.0
    )

    detect_submitted = st.form_submit_button(
        "Predict Cancer Status",
        use_container_width=True
    )

if "cancer_prediction" not in st.session_state:
    st.session_state.cancer_prediction = None

if detect_submitted:
    patient_data = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "M" else 0,
        "plasma_CA19_9": plasma_CA19_9,
        "creatinine": creatinine,
        "LYVE1": LYVE1,
        "REG1B": REG1B,
        "TFF1": TFF1,
        "REG1A": REG1A
    }])

    st.session_state.cancer_prediction = detection_model.predict(
        patient_data
    )[0]

    # Clear any previous stage result for the new prediction.
    st.session_state.stage_prediction = None

prediction = st.session_state.cancer_prediction

if prediction is not None:
    st.subheader("Cancer Detection Result")

    if prediction == "Cancer":
        st.error("Model Prediction: Cancer")
        st.info(
            "This is only a model output, not a confirmed diagnosis. "
            "The stage form below is a separate educational demonstration."
        )
    else:
        st.success("Model Prediction: No Cancer")
        st.info(
            "This result does not rule out cancer. Consult a qualified "
            "healthcare professional for medical concerns."
        )

# -----------------------------------
# STEP 2: CANCER STAGE PREDICTION
# -----------------------------------

if prediction == "Cancer":
    st.divider()
    st.header("Step 2: Cancer Stage Prediction")

    st.write(
        "Enter the features required by the existing stage model. "
        "The available choices and ranges come from its training dataset."
    )

    excluded_columns = [
        "Country",
        "Stage_at_Diagnosis",
        "Survival_Time_Months",
        "Survival_Status",
        "Treatment_Type"
    ]

    stage_features = stage_df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    stage_input = {}

    with st.form("stage_form"):
        st.subheader("Patient Details")

        for column in stage_features.columns:
            series = stage_features[column].dropna()
            
            if series.empty:
                st.warning(f"No training values available for {column}.")
                continue

            if pd.api.types.is_numeric_dtype(series):
                unique_values = sorted(series.unique().tolist())

                if len(unique_values) <= 2:
                    options = unique_values
                    default_index = 0
                    stage_input[column] = st.selectbox(
                        column,
                        options=options,
                        index=default_index,
                        key=f"stage_{column}"
                    )
                else:
                    minimum = float(series.min())
                    maximum = float(series.max())
                    median = float(series.median())

                    if pd.api.types.is_integer_dtype(series):
                        stage_input[column] = st.number_input(
                            column,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(median),
                            step=1,
                            key=f"stage_{column}"
                        )
                    else:
                        stage_input[column] = st.number_input(
                            column,
                            min_value=minimum,
                            max_value=maximum,
                            value=median,
                            key=f"stage_{column}"
                        )
            else:
                options = sorted(series.astype(str).unique().tolist())
                stage_input[column] = st.selectbox(
                    column,
                    options=options,
                    key=f"stage_{column}"
                )

        stage_submitted = st.form_submit_button(
            "Predict Cancer Stage",
            use_container_width=True
        )

    if stage_submitted:
        stage_patient = pd.DataFrame(
            [stage_input],
            columns=stage_features.columns
        )

        try:
            stage_result = stage_model.predict(stage_patient)[0]
            st.session_state.stage_prediction = stage_result
        except Exception as e:
            st.error(f"Could not predict the cancer stage: {e}")

    if st.session_state.get("stage_prediction") is not None:
        st.subheader("Stage Prediction Result")
        st.success(
            f"Model Prediction: {st.session_state.stage_prediction}"
        )
        st.caption(
            "This stage prediction is generated by a separate model trained "
            "on a different dataset. It is not a clinical staging result."
        )

st.divider()
st.caption(
    "Machine Learning Project | Python | Scikit-learn | Streamlit"
)
