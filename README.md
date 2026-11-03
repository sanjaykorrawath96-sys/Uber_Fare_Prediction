# Uber_Fare_Prediction

# 🚖 Uber Fare Prediction using Machine Learning

A machine learning project that predicts the fare of a taxi / ride-hailing trip **before the ride starts**, using trip distance, duration, passenger count, rate type and pickup time. The final model is deployed as an interactive **Streamlit** app that can also compare fares across morning, afternoon, evening and night.

> **Note:** The project uses NYC Yellow Taxi trip records (March 2016) as a proxy for Uber trip data.



---

## 📌 Business Problem

Uber and its competitors (Lyft, Ola, Bolt and others) must quote an accurate fare upfront, but the fare depends on many interacting factors such as distance, duration, pickup time, day and rate type.

- **Under-priced estimates** cause revenue loss for Uber and drivers, or a surprise final price for the rider.
- **Over-priced estimates** make riders cancel or switch to a competitor.
- **Inconsistent estimates** reduce customer trust, increase complaints, and make demand planning and driver incentives harder.

## 🎯 Objective

Build a machine learning model that estimates accurate trip fares before the ride begins, so that platforms can offer transparent upfront pricing, retain customers and protect driver earnings.

---

## 📊 Dataset

- **Source:** NYC Taxi and Limousine Commission (TLC) Yellow Taxi trip records, March 2016
- **Size:** 19 raw columns, [update: number of rows used] trips
- **Target variable:** `fare_amount` (USD)
- **Data quality:** no missing values or duplicates; invalid records (negative fares, zero durations, extreme distances) were removed

| Column group | Columns | Meaning |
|---|---|---|
| Trip details | `passenger_count`, `trip_distance` | Passengers and distance in miles |
| Rate type | `RatecodeID` | 1 Standard, 2 JFK, 3 Newark, 4 Nassau/Westchester, 5 Negotiated, 6 Group ride |
| Time details | `tpep_pickup_datetime`, `tpep_dropoff_datetime` | Trip start and end time |
| Geographic details | pickup/dropoff longitude and latitude | GPS coordinates |
| Vendor and system | `VendorID`, `store_and_fwd_flag` | Technology provider and record-storage flag |
| Payment details | `payment_type` | How the passenger paid |
| Fixed charges | `extra`, `mta_tax`, `improvement_surcharge` | Surcharges and taxes |
| Post-trip charges | `tip_amount`, `tolls_amount`, `total_amount` | Known only after the trip (leakage) |
| **Target** | `fare_amount` | Metered fare |

---

## 🔄 Methodology (CRISP-ML(Q))

1. **Business and data understanding:** define the goal, success metrics and data quality.
2. **Data engineering:**
   - Cleaned invalid fares, durations, distances and rate codes.
   - Created `duration_minutes`, `day`, `day_of_week`, `is_weekend`, `hour` and `is_night`.
   - Dropped leakage columns (`total_amount`, `tip_amount`, `tolls_amount`) and low-value columns.
3. **Model engineering:**
   - Pipeline: `StandardScaler` → `PowerTransformer` (Yeo-Johnson) on all 9 model inputs.
   - Compared Linear Regression, Decision Tree, Random Forest, XGBoost and SVR.
   - Tuned the best model (Random Forest) with **Optuna** (50 trials).
4. **Quality assurance:** evaluated with R², MAE and RMSE; checked overfitting using train vs. test scores and 5-fold cross-validation.
5. **Deployment:** saved the model as a pickle file and built a Streamlit app.
6. **Monitoring and maintenance:** planned (track error on new trips, retrain, add new data sources).

**Model inputs (9):** `passenger_count`, `trip_distance`, `RatecodeID`, `duration_minutes`, `day`, `day_of_week`, `is_weekend`, `hour`, `is_night`

---

## 📈 Results

| Model | Test R² | MAE ($) | RMSE ($) |
|---|---|---|---|
| Linear Regression | 0.818 | 2.55 | 4.38 |
| Decision Tree | 0.916 | 0.47 | 3.04 |
| XGBoost | 0.955 | 0.43 | 2.23 |
| Random Forest | 0.970 | 0.39 | 1.82 |
| **Random Forest (tuned with Optuna)** | **0.972** | **0.37** | **1.75** |

**Best hyperparameters:** `n_estimators=175`, `max_depth=14`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features=None`

**Key insights**
- Raw data contained outliers (negative fares, extreme distances), so cleaning was essential.
- Dropping post-trip columns prevents data leakage and keeps the model realistic for upfront pricing.
- Fare is non-linear: tree-based models clearly beat Linear Regression.
- The Decision Tree overfit (train R² ≈ 0.9999 vs. test R² ≈ 0.918); Random Forest was more stable in cross-validation (mean R² ≈ 0.944 vs. 0.927 for XGBoost).
- The final model explains about **97%** of fare variance, with a typical error of about **$0.37** per trip.

---

## 🖥️ Streamlit App

The app has two tabs:

- **Single prediction:** enter distance, duration, passengers, rate type, date and time to get an estimated fare.
- **Compare times of day:** predicts the same trip at Morning (9 AM), Afternoon (2 PM), Evening (6 PM) and Night (10 PM) and shows the fares in metric cards, a bar chart and a comparison table. An optional setting lets you use a different duration per period to reflect traffic.

<!-- Add a screenshot of your running app here -->
<!-- ![App screenshot](app_screenshot.png) -->

---

## 🗂️ Project Structure

```
├── Uber_Price_Prediction.ipynb    # EDA, preprocessing, modelling, tuning
├── app.py                         # Streamlit app
├── uber_banner.png                # Banner image used in the app
├── Uber Price prediction.pkl      # Trained model (see note below)
├── requirements.txt               # Python dependencies
└── README.md
```

> If the model file is too large for GitHub (100 MB limit), use [Git LFS](https://git-lfs.com/), upload it to cloud storage, or retrain by running the notebook.

---

## ⚙️ Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

To retrain the model, open `Uber_Price_Prediction.ipynb`, run all cells and save the resulting pickle file next to `app.py`. Use the same scikit-learn version for training and for running the app.

---

## ⚠️ Limitations and Future Scope

- The data covers a single month of NYC yellow taxi trips (not actual Uber data), and the rows used come from the start of the file, so weekend trips are barely or not represented. **Random sampling across the full month** is the next step.
- Surge pricing, traffic, weather and events are not captured.
- Pickup and dropoff coordinates were not used; adding zone and airport features could improve accuracy.
- `RatecodeID` is treated as numeric; one-hot encoding is a possible improvement.
- Future work: train on more months, add monitoring, and deploy as a web or mobile service.

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Optuna, Streamlit



---

⭐ If you found this project useful, consider giving it a star.
