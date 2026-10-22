
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Manufacturing Output Prediction",
    page_icon="🏭",
    layout="wide"
)


# --------------------------------------------------
# Default Input Values
# --------------------------------------------------

DEFAULT_VALUES = {
    "injection_temperature": 200.0,
    "injection_pressure": 100.0,
    "cycle_time": 30.0,
    "cooling_time": 10.0,
    "material_viscosity": 100.0,
    "ambient_temperature": 25.0,
    "machine_age": 5.0,
    "operator_experience": 5.0,
    "maintenance_hours": 10.0,
    "efficiency_score": 75.0,
    "machine_utilization": 75.0,
    "temperature_pressure_ratio": 2.0,
    "total_cycle_time": 40.0,
    "hour": 12,
    "shift": "Day",
    "machine_type": "Type_A",
    "material_grade": "Economy",
    "day_of_week": "Monday",
}


# Initialize input values when the app first runs.
for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Reset Input Values
# --------------------------------------------------

def reset_inputs():
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("manufacturing_output_model.pkl")


model = load_model()


# --------------------------------------------------
# Title and Description
# --------------------------------------------------

st.title("🏭 Manufacturing Equipment Output Prediction")

st.markdown(
    """
    This application predicts the expected number of plastic parts produced
    per hour based on machine operating conditions, material properties,
    environmental conditions, and production settings.
    """
)

st.divider()


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Enter Manufacturing Parameters")

col1, col2, col3 = st.columns(3)


# --------------------------------------------------
# Numerical Inputs
# --------------------------------------------------

with col1:

    injection_temperature = st.number_input(
        "Injection Temperature",
        min_value=0.0,
        step=1.0,
        key="injection_temperature"
    )

    injection_pressure = st.number_input(
        "Injection Pressure",
        min_value=0.0,
        step=1.0,
        key="injection_pressure"
    )

    cycle_time = st.number_input(
        "Cycle Time",
        min_value=0.0,
        step=0.1,
        key="cycle_time"
    )

    cooling_time = st.number_input(
        "Cooling Time",
        min_value=0.0,
        step=0.1,
        key="cooling_time"
    )

    material_viscosity = st.number_input(
        "Material Viscosity",
        min_value=0.0,
        step=1.0,
        key="material_viscosity"
    )

    ambient_temperature = st.number_input(
        "Ambient Temperature",
        step=0.1,
        key="ambient_temperature"
    )


with col2:

    machine_age = st.number_input(
        "Machine Age",
        min_value=0.0,
        step=1.0,
        key="machine_age"
    )

    operator_experience = st.number_input(
        "Operator Experience",
        min_value=0.0,
        step=1.0,
        key="operator_experience"
    )

    maintenance_hours = st.number_input(
        "Maintenance Hours",
        min_value=0.0,
        step=1.0,
        key="maintenance_hours"
    )

    efficiency_score = st.number_input(
        "Efficiency Score",
        min_value=0.0,
        step=0.1,
        key="efficiency_score"
    )

    machine_utilization = st.number_input(
        "Machine Utilization",
        min_value=0.0,
        step=0.1,
        key="machine_utilization"
    )

    temperature_pressure_ratio = st.number_input(
        "Temperature Pressure Ratio",
        min_value=0.0,
        step=0.01,
        key="temperature_pressure_ratio"
    )


with col3:

    total_cycle_time = st.number_input(
        "Total Cycle Time",
        min_value=0.0,
        step=0.1,
        key="total_cycle_time"
    )

    hour = st.number_input(
        "Hour of Day",
        min_value=0,
        max_value=23,
        step=1,
        key="hour"
    )


# --------------------------------------------------
# Categorical Inputs
# --------------------------------------------------

st.subheader("Production Settings")

cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

with cat_col1:

    shift = st.selectbox(
        "Shift",
        ["Day", "Evening", "Night"],
        key="shift"
    )

with cat_col2:

    machine_type = st.selectbox(
        "Machine Type",
        ["Type_A", "Type_B", "Type_C"],
        key="machine_type"
    )

with cat_col3:

    material_grade = st.selectbox(
        "Material Grade",
        ["Economy", "Standard", "Premium"],
        key="material_grade"
    )

with cat_col4:

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ],
        key="day_of_week"
    )


st.divider()


# --------------------------------------------------
# Prediction and Reset Buttons
# --------------------------------------------------

predict_col, reset_col = st.columns(2)

with predict_col:
    predict_clicked = st.button(
        "🔮 Predict Parts Per Hour",
        use_container_width=True
    )

with reset_col:
    st.button(
        "🔄 Reset Inputs",
        on_click=reset_inputs,
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_clicked:

    # Create input dataframe using the same feature names
    # used during model training.

    input_data = pd.DataFrame({
        "Injection_Temperature": [injection_temperature],
        "Injection_Pressure": [injection_pressure],
        "Cycle_Time": [cycle_time],
        "Cooling_Time": [cooling_time],
        "Material_Viscosity": [material_viscosity],
        "Ambient_Temperature": [ambient_temperature],
        "Machine_Age": [machine_age],
        "Operator_Experience": [operator_experience],
        "Maintenance_Hours": [maintenance_hours],
        "Shift": [shift],
        "Machine_Type": [machine_type],
        "Material_Grade": [material_grade],
        "Day_of_Week": [day_of_week],
        "Temperature_Pressure_Ratio": [temperature_pressure_ratio],
        "Total_Cycle_Time": [total_cycle_time],
        "Efficiency_Score": [efficiency_score],
        "Machine_Utilization": [machine_utilization],
        "Hour": [hour]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display Prediction
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col2:
        st.metric(
            label="Predicted Parts Per Hour",
            value=f"{prediction:.2f}"
        )

    st.success(
        f"The predicted production output is "
        f"**{prediction:.2f} parts per hour**."
    )

    # Show Input Summary
    with st.expander("View Input Parameters"):
        st.dataframe(
            input_data.T.rename(columns={0: "Value"}),
            use_container_width=True
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Manufacturing Equipment Output Prediction | "
    "Linear Regression Model"
)
