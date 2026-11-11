import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Smart Electricity Predictor",
    page_icon="⚡",
    layout="wide"
)

# Model loading
model = joblib.load("model/ann_model.pkl")


# --------------------------------
# DELHI BRPL BILL CALCULATOR
# --------------------------------

def calculate_bill(units):

    # Delhi domestic energy tariff slabs
    if units <= 200:
        energy_charge = units * 3.00

    elif units <= 400:
        energy_charge = (
            200 * 3.00
            + (units - 200) * 4.50
        )

    elif units <= 800:
        energy_charge = (
            200 * 3.00
            + 200 * 4.50
            + (units - 400) * 6.50
        )

    elif units <= 1200:
        energy_charge = (
            200 * 3.00
            + 200 * 4.50
            + 400 * 6.50
            + (units - 800) * 7.00
        )

    else:
        energy_charge = (
            200 * 3.00
            + 200 * 4.50
            + 400 * 6.50
            + 400 * 7.00
            + (units - 1200) * 8.00
        )

    # Published Delhi subsidy scheme
    if units <= 200:
        subsidy = energy_charge

    elif units <= 400:
        subsidy = min(energy_charge, 800)

    else:
        subsidy = 0

    final_bill = max(0, energy_charge - subsidy)

    return energy_charge, subsidy, final_bill


# --------------------------------
# TITLE
# --------------------------------

st.title("⚡ Smart Electricity Bill Predictor")

st.write(
    "Predict your monthly electricity consumption "
    "using Artificial Neural Networks."
)

st.caption("Electricity Provider: BSES Rajdhani Power Limited (BRPL), Delhi")

st.divider()


# --------------------------------
# HOUSEHOLD INPUTS
# --------------------------------

st.subheader("Enter Household Details")

col1, col2 = st.columns(2)


# LEFT COLUMN
with col1:

    family_members = st.number_input(
        "Number of Family Members", 1, 15, 4
    )

    ac_count = st.number_input(
        "Number of ACs", 0, 10, 1
    )

    # AC Usage
    ac_col1, ac_col2 = st.columns(2)

    with ac_col1:
        ac_hour = st.number_input(
            "AC Hours", 0, 24, 5
        )

    with ac_col2:
        ac_minute = st.selectbox(
            "AC Minutes", [0, 10, 20, 30, 40, 50]
        )

    ac_hours = ac_hour + ac_minute / 60

    # Fan Usage
    fan_count = st.number_input(
        "Number of Fans", 0, 20, 4
    )

    fan_col1, fan_col2 = st.columns(2)

    with fan_col1:
        fan_hour = st.number_input(
            "Fan Hours", 0, 24, 10
        )

    with fan_col2:
        fan_minute = st.selectbox(
            "Fan Minutes", [0, 10, 20, 30, 40, 50]
        )

    fan_hours = fan_hour + fan_minute / 60

    fridge_count = st.number_input(
        "Number of Refrigerators", 0, 5, 1
    )

    washing_loads = st.number_input(
        "Weekly Washing Machine Loads", 0, 30, 4
    )


# RIGHT COLUMN
with col2:

    tv_count = st.number_input(
        "Number of TVs", 0, 10, 1
    )

    # TV Usage
    tv_col1, tv_col2 = st.columns(2)

    with tv_col1:
        tv_hour = st.number_input(
            "TV Hours", 0, 24, 4
        )

    with tv_col2:
        tv_minute = st.selectbox(
            "TV Minutes", [0, 10, 20, 30, 40, 50]
        )

    tv_hours = tv_hour + tv_minute / 60

    lights_count = st.number_input(
        "Number of Lights", 0, 30, 6
    )

    # Light Usage
    light_col1, light_col2 = st.columns(2)

    with light_col1:
        light_hour = st.number_input(
            "Light Hours", 0, 24, 6
        )

    with light_col2:
        light_minute = st.selectbox(
            "Light Minutes", [0, 10, 20, 30, 40, 50]
        )

    light_hours = light_hour + light_minute / 60


st.divider()


# --------------------------------
# ANN PREDICTION
# --------------------------------

if st.button("⚡ Predict Electricity Consumption"):

    input_data = pd.DataFrame([{
        "family_members": family_members,
        "ac_count": ac_count,
        "ac_hours": ac_hours,
        "fan_count": fan_count,
        "fan_hours": fan_hours,
        "fridge_count": fridge_count,
        "washing_loads": washing_loads,
        "tv_count": tv_count,
        "tv_hours": tv_hours,
        "lights_count": lights_count,
        "light_hours": light_hours
    }])

    # ANN prediction
    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    # --------------------------------
    # BILL CALCULATION
    # --------------------------------

    energy_charge, subsidy, bill = calculate_bill(prediction)

    # --------------------------------
    # PREDICTION RESULTS
    # --------------------------------

    st.success("Prediction Completed!")

    st.subheader("Electricity Bill Estimation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Monthly Consumption",
            f"{prediction:.2f} Units"
        )

    with col2:
        st.metric(
            "Electricity Subsidy",
            f"₹{subsidy:.2f}"
        )

    with col3:
        st.metric(
            "Estimated Energy Bill",
            f"₹{bill:.2f}"
        )

    st.write(f"Energy Charges: ₹{energy_charge:.2f}")

    st.caption(
        "Bill estimate uses published Delhi domestic energy "
        "tariff slabs and assumes subsidy eligibility. "
        "Fixed charges, PPAC, electricity tax and other "
        "applicable charges are excluded. The subsidy scheme "
        "should be verified for the current financial year."
    )

    # --------------------------------
    # CONSUMPTION ANALYSIS
    # --------------------------------

    st.subheader("Consumption Analysis")

    if prediction < 150:
        st.info("Low electricity consumption.")

    elif prediction < 350:
        st.info("Moderate electricity consumption.")

    else:
        st.warning("High electricity consumption.")

    # --------------------------------
    # GRAPH 1: MONTHLY CONSUMPTION
    # --------------------------------

    st.divider()

    st.subheader("1. Monthly Electricity Consumption")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        ["Predicted Consumption"],
        [prediction]
    )

    ax.axhline(
        150,
        linestyle="--",
        label="Low / Moderate boundary (150 Units)"
    )

    ax.axhline(
        350,
        linestyle="--",
        label="Moderate / High boundary (350 Units)"
    )

    ax.set_ylabel("Electricity Consumption (kWh)")
    ax.set_title("ANN Predicted Monthly Consumption")

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------
    # APPLIANCE-WISE ESTIMATES
    # --------------------------------

    appliance_usage = {

        "AC": ac_count * ac_hours * 30 * 1.2,

        "Fans": fan_count * fan_hours * 30 * 0.075,

        "Refrigerator": fridge_count * 1.2 * 30,

        "Washing Machine": washing_loads * 4.3 * 0.7,

        "TV": tv_count * tv_hours * 30 * 0.1,

        "Lights": lights_count * light_hours * 30 * 0.012
    }

    appliance_df = pd.DataFrame(
        list(appliance_usage.items()),
        columns=["Appliance", "Estimated kWh"]
    )

    # --------------------------------
    # GRAPH 2: APPLIANCE BAR CHART
    # --------------------------------

    st.divider()

    st.subheader("2. Appliance-wise Electricity Consumption")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        appliance_df["Appliance"],
        appliance_df["Estimated kWh"]
    )

    ax.set_xlabel("Electrical Appliances")
    ax.set_ylabel("Estimated Consumption (kWh)")

    ax.set_title("Estimated Monthly Appliance Consumption")

    plt.xticks(rotation=25)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------
    # GRAPH 3: DONUT CHART
    # --------------------------------

    st.divider()

    st.subheader("3. Appliance Consumption Distribution")

    if appliance_df["Estimated kWh"].sum() > 0:

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.pie(
            appliance_df["Estimated kWh"],
            labels=appliance_df["Appliance"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.4}
        )

        ax.set_title("Appliance-wise Energy Distribution")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No appliance consumption to display.")

    st.caption(
        "The ANN predicts total monthly consumption. "
        "Appliance-wise graphs use illustrative power-rating "
        "assumptions and are estimates, not separate ANN predictions. "
        "Their total may differ from the ANN prediction."
    )

    # --------------------------------
    # ENERGY SAVING SUGGESTIONS
    # --------------------------------

    st.divider()

    st.subheader("💡 Electricity Saving Suggestions")

    if ac_count > 0:
        st.write("- Use AC at an energy-efficient temperature.")

    if fan_count > 0:
        st.write("- Switch off fans when not required.")

    if light_hours > 8:
        st.write("- Reduce unnecessary lighting hours.")

    st.write("- Prefer energy-efficient appliances.")
