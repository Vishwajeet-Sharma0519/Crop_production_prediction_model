import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load('production_prediction_pipeline.pkl')
    except Exception as e1:
        # Fallback if xgboost pipeline isn't there, try another
        try:
            return joblib.load('best_crop_production_model.pkl') 
        except Exception as e2:
            st.error(f"Failed to load model. Error 1: {e1} | Error 2: {e2}")
            st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv('final_crop_yield_dataset.csv')
    return df

st.title("Crop Yield Prediction App")
st.markdown("Enter the agricultural details below to predict the crop production yield.")

# Load resources
model = load_model()
data = load_data()

# Create layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Location Details")
    state = st.selectbox("State", sorted(data['State_Name'].unique()))
    
    # Filter districts based on selected state
    districts_in_state = data[data['State_Name'] == state]['District_Name'].unique()
    district = st.selectbox("District", sorted(districts_in_state))
    
    area = st.number_input("Area (in Hectares)", min_value=1.0, value=15000.0, step=100.0)

with col2:
    st.subheader("Crop Details")
    season = st.selectbox("Season", sorted(data['Season'].unique()))
    crop = st.selectbox("Crop", sorted(data['Crop'].unique()))
    crop_year = st.number_input("Crop Year", min_value=1990, max_value=2050, value=2024, step=1)

st.markdown("---")
st.subheader("Weather Conditions")
weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

with weather_col1:
    rainfall = st.number_input("Rainfall (Jan-Aug mm)", min_value=0.0, value=500.5, step=10.0)
with weather_col2:
    tavg = st.number_input("Avg Temp (°C)", min_value=-10.0, value=28.5, step=0.5)
with weather_col3:
    tmin = st.number_input("Min Temp (°C)", min_value=-20.0, value=22.0, step=0.5)
with weather_col4:
    tmax = st.number_input("Max Temp (°C)", min_value=-10.0, value=35.0, step=0.5)
    
prcp = st.number_input("Daily Precipitation (mm)", min_value=0.0, value=15.2, step=1.0)

# Prediction Button
st.markdown("---")
if st.button("Predict Crop Production", type="primary", use_container_width=True):
    # Construct the input dataframe
    input_data = pd.DataFrame({
        'State_Name': [state],
        'District_Name': [district],
        'Crop_Year': [crop_year],
        'Season': [season],
        'Crop': [crop],
        'Area': [area],
        'Total_Rainfall_Jan_Aug': [rainfall],
        'tavg': [tavg],
        'tmin': [tmin],
        'tmax': [tmax],
        'prcp': [prcp]
    })
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        pred_value = float(prediction[0])
        
        if pred_value < 0:
            st.warning("The predicted production is effectively zero. It is highly likely that your entered region or season does not suit this crop!")
            st.metric(label="Estimated Crop Production (in Tonnes)", value="0.00 Tonnes")
            st.info("Estimated Yield: 0.0000 Tonnes per Hectare")
        else:
            st.success("Prediction Successful!")
            st.metric(label="Estimated Crop Production (in Tonnes)", value=f"{pred_value:,.2f} Tonnes")
            
            # Display Yield (Production / Area)
            st.info(f"Estimated Yield: {pred_value/area:,.4f} Tonnes per Hectare")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
