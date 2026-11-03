import pickle
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- Page setup ----------

BANNER_IMAGE_PATH = "banner.png"

st.image(r"C:\Users\Sanju\Downloads\Uber Night Ride in the City.png", use_container_width=True)

st.set_page_config(page_title="Uber Fare Predictor", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Uber_Price_Prediction.pkl"
BANNER_PATH = BASE_DIR / "uber_banner.png"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH.name}. Place it in the same folder as app.py.")
    st.stop()

RATE_CODES = {
    "Standard rate": 1,
    "JFK": 2,
    "Newark": 3,
    "Nassau or Westchester": 4,
    "Negotiated fare": 5,
    "Group ride": 6,
}

# Representative pickup hour for each part of the day
PERIODS = {
    "Morning (9 AM)": 9,
    "Afternoon (2 PM)": 14,
    "Evening (6 PM)": 18,
    "Night (10 PM)": 22,
}


def build_features(distance, duration, passengers, rate_code, pickup_dt):
    """Create the same 9 features the model was trained on."""
    hour = pickup_dt.hour
    dow = pickup_dt.weekday()  # 0 = Monday ... 6 = Sunday
    return pd.DataFrame(
        [
            {
                "passenger_count": passengers,
                "trip_distance": distance,
                "RatecodeID": rate_code,
                "duration_minutes": duration,
                "day": pickup_dt.day,
                "day_of_week": dow,
                "is_weekend": int(dow in (5, 6)),
                "hour": hour,
                "is_night": int(hour >= 20 or hour < 6),
            }
        ]
    )


# ---------- Header ----------
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), width="stretch")

st.title("Machine Learning-Powered Uber Fare Price Predictor")


# ---------- Shared trip inputs ----------
col1, col2 = st.columns(2)
with col1:
    trip_distance = st.number_input("Trip distance (miles)", min_value=0.1, max_value=300.0, value=3.0, step=0.1)
    passenger_count = st.number_input("Passengers", min_value=1, max_value=6, value=1, step=1)
    rate_label = st.selectbox("Rate type", list(RATE_CODES.keys()))
with col2:
    duration_minutes = st.number_input("Trip duration (minutes)", min_value=1.0, max_value=600.0, value=15.0, step=1.0)
    pickup_date = st.date_input("Pickup date", value=datetime.today())
rate_code = RATE_CODES[rate_label]

tab1, tab2 = st.tabs(["Single prediction", "Compare: Morning / Afternoon / Evening / Night"])

# ---------- Tab 1: single prediction ----------
with tab1:
    pickup_time = st.time_input("Pickup time", value=time(18, 30))
    if st.button("Predict fare", type="primary", width="stretch", key="single"):
        try:
            feats = build_features(
                trip_distance, duration_minutes, passenger_count, rate_code,
                datetime.combine(pickup_date, pickup_time),
            )
            fare = float(model.predict(feats)[0])
            st.success(f"Estimated fare: **${fare:,.2f}**")
            with st.expander("Features sent to the model"):
                st.dataframe(feats, hide_index=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- Tab 2: compare times of day ----------
with tab2:
    st.write("Same trip, same date, predicted at four times of the day.")

    use_custom = st.checkbox("Use a different trip duration for each period (e.g. to reflect traffic)")
    durations = {}
    if use_custom:
        cols = st.columns(4)
        for c, name in zip(cols, PERIODS):
            durations[name] = c.number_input(name.split(" ")[0], min_value=1.0, max_value=600.0,
                                              value=float(duration_minutes), step=1.0, key=f"dur_{name}")
    else:
        durations = {name: duration_minutes for name in PERIODS}

    if st.button("Compare fares", type="primary", width="stretch", key="compare"):
        try:
            rows = []
            for name, hour in PERIODS.items():
                pickup_dt = datetime.combine(pickup_date, time(hour, 0))
                feats = build_features(trip_distance, durations[name], passenger_count, rate_code, pickup_dt)
                rows.append({"Period": name.split(" ")[0], "Pickup": name.split("(")[1].rstrip(")"),
                             "Duration (min)": durations[name], "Predicted fare ($)": float(model.predict(feats)[0])})

            result = pd.DataFrame(rows)
            base = result["Predicted fare ($)"].iloc[0]
            result["Difference vs Morning ($)"] = result["Predicted fare ($)"] - base
            result["Difference vs Morning (%)"] = result["Difference vs Morning ($)"] / base * 100

            # Metric cards
            cols = st.columns(4)
            for c, (_, r) in zip(cols, result.iterrows()):
                c.metric(r["Period"], f"${r['Predicted fare ($)']:.2f}",
                         None if r["Period"] == "Morning" else f"{r['Difference vs Morning ($)']:+.2f}")

            # Chart
            st.bar_chart(result.set_index("Period")["Predicted fare ($)"])

            # Table
            st.dataframe(
                result.style.format({
                    "Duration (min)": "{:.0f}",
                    "Predicted fare ($)": "{:.2f}",
                    "Difference vs Morning ($)": "{:+.2f}",
                    "Difference vs Morning (%)": "{:+.1f}%",
                }),
                hide_index=True, width="stretch",
            )

            cheapest = result.loc[result["Predicted fare ($)"].idxmin(), "Period"]
            costliest = result.loc[result["Predicted fare ($)"].idxmax(), "Period"]
            st.info(f"Lowest fare: **{cheapest}** | Highest fare: **{costliest}** "
                    f"(spread ${result['Predicted fare ($)'].max() - result['Predicted fare ($)'].min():.2f}).")
        except Exception as e:
            st.error(f"Prediction failed: {e}")