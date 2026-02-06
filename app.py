"""
===============================================================================
STREAMLIT APP - app.py
===============================================================================
Save this file as: app.py
Place it in the same folder as your .pkl files
Run with: streamlit run app.py
"""

import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        # Load Random Forest model (saved with joblib)
        model = joblib.load('car_price_rf_model.pkl')
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load categorical info (optional)
        try:
            with open('categorical_info.pkl', 'rb') as f:
                categorical_info = pickle.load(f)
        except:
            categorical_info = {}
        
        return model, scaler, label_encoders, feature_names, categorical_info
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Could not find {e.filename}")
        st.info("""
        **Missing files!** Please make sure these files are in the same folder as app.py:
        - rf_model.pkl
        - scaler.pkl
        - label_encoders.pkl
        - feature_names.pkl
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

model, scaler, label_encoders, feature_names, categorical_info = load_models()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<h1 class="main-header">🚗 Used Car Price Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
    <h3>Predict the market value of used cars using Machine Learning</h3>
    <p>This AI model uses <strong>Random Forest Regressor</strong> trained on <strong>188,533 vehicles</strong> 
    to provide accurate price predictions.</p>
    <p><strong>Model Performance:</strong> Kaggle Private Score: <span style='color: #28a745; font-weight: bold;'>66,024 RMSE</span></p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - USER INPUTS
# ============================================================

st.sidebar.header("🔧 Vehicle Specifications")
st.sidebar.markdown("Fill in the details about the vehicle")

# Get available options from categorical_info or use defaults
brands = categorical_info.get('brand', [
    'Ford', 'Mercedes-Benz', 'BMW', 'Chevrolet', 'Audi', 'Toyota', 'Honda', 
    'Nissan', 'Hyundai', 'Kia', 'Volkswagen', 'Mazda', 'Subaru', 'Lexus',
    'Porsche', 'Tesla', 'Jaguar', 'Land Rover', 'Volvo', 'Jeep', 'RAM',
    'Cadillac', 'Acura', 'Infiniti', 'Genesis', 'Buick', 'GMC', 'Dodge'
])

fuel_types = categorical_info.get('fuel_type', [
    'Gasoline', 'Hybrid', 'Diesel', 'Electric', 'Plug-In Hybrid', 'E85 Flex Fuel'
])

transmissions = categorical_info.get('transmission', [
    'A/T', '8-Speed A/T', '6-Speed A/T', '7-Speed A/T', '9-Speed A/T', 
    '10-Speed A/T', 'M/T', 'CVT', 'Transmission w/Dual Shift Mode'
])

# Basic Information
st.sidebar.subheader("📋 Basic Information")

brand = st.sidebar.selectbox("Brand", options=sorted(brands), index=0)
model_input = st.sidebar.text_input("Model", value="Civic", help="Enter vehicle model name")
model_year = st.sidebar.slider("Model Year", min_value=1990, max_value=2024, value=2020, step=1)
mileage = st.sidebar.number_input("Mileage (miles)", min_value=100, max_value=400000, value=50000, step=1000)

# Specifications
st.sidebar.subheader("⚙️ Specifications")

fuel_type = st.sidebar.selectbox("Fuel Type", options=fuel_types)
transmission = st.sidebar.selectbox("Transmission", options=transmissions)

# Engine Details
st.sidebar.subheader("🔧 Engine Details")

horsepower = st.sidebar.number_input("Horsepower (HP)", min_value=75, max_value=1000, value=250, step=10)
engine_displacement = st.sidebar.number_input("Engine Displacement (L)", 
                                               min_value=0.0, max_value=8.0, value=2.5, step=0.1)
cylinders = st.sidebar.selectbox("Number of Cylinders", options=[3, 4, 5, 6, 8, 10, 12], index=3)

# Condition
st.sidebar.subheader("📊 Condition")

accident_options = ['None reported', 'At least 1 accident or damage reported']
accident = st.sidebar.radio("Accident History", options=accident_options)

clean_title_options = ['Yes', 'No']
clean_title = st.sidebar.radio("Clean Title", options=clean_title_options)

# Colors
st.sidebar.subheader("🎨 Colors")

ext_color = st.sidebar.selectbox("Exterior Color", 
    options=['Black', 'White', 'Silver', 'Gray', 'Blue', 'Red', 'Green', 'Brown', 'Orange', 'Yellow', 'Purple'])

int_color = st.sidebar.selectbox("Interior Color", 
    options=['Black', 'Beige', 'Gray', 'Brown', 'White', 'Red', 'Tan'])

# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(data):
    """Apply same feature engineering as training"""
    
    # Current year for age calculation
    CURRENT_YEAR = 2025
    
    # Car age
    data['car_age'] = CURRENT_YEAR - data['model_year']
    
    # Mileage per year
    data['mileage_per_year'] = data['milage'] / (data['car_age'] + 1)
    
    # Luxury brand indicator (must match training notebook)
    luxury_brands = [
        'BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche',
        'Tesla', 'Land Rover', 'Jaguar', 'Cadillac', 'Lincoln',
        'Maserati', 'Bentley', 'Rolls-Royce', 'Lamborghini', 'Ferrari'
    ]
    data['is_luxury'] = data['brand'].isin(luxury_brands).astype(int)
    
    # Performance metrics
    data['hp_per_cylinder'] = data['horsepower'] / (data['cylinder_count'] + 1)
    data['is_high_performance'] = (
        (data['horsepower'] > 300) | (data['cylinder_count'] >= 8)
    ).astype(int)
    
    # Depreciation indicator
    data['depreciation_indicator'] = data['milage'] * data['car_age']
    
    # Binary condition flags
    data['has_accident'] = (
        data['accident'] == 'At least 1 accident or damage reported'
    ).astype(int)
    
    data['is_clean_title'] = (data['clean_title'] == 'Yes').astype(int)
    data['has_clean_title'] = data['is_clean_title']  # Alias
    
    return data

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(input_data):
    """Process input data and make prediction"""
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categorical variables (store as col_encoded to match training)
    categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col']

    for col in categorical_cols:
        encoded_col = col + '_encoded'
        if col in label_encoders and col in df.columns:
            try:
                if df[col].iloc[0] in label_encoders[col].classes_:
                    df[encoded_col] = label_encoders[col].transform(df[col])
                else:
                    df[encoded_col] = 0
            except Exception as e:
                df[encoded_col] = 0
        else:
            df[encoded_col] = 0

    # Ensure all required features exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in correct order
    df_features = df[feature_names]
    
    # Scale features
    df_scaled = scaler.transform(df_features)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    
    return max(0, prediction)  # Ensure non-negative price

# ============================================================
# MAIN CONTENT AREA
# ============================================================

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📊 About the Model", "❓ Help"])

with tab1:
    st.header("Price Prediction Results")
    
    # Predict Button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        
        # Prepare input data
        input_data = {
            'brand': brand,
            'model': model_input,
            'model_year': model_year,
            'milage': mileage,
            'fuel_type': fuel_type,
            'transmission': transmission,
            'horsepower': horsepower,
            'engine_displacement': engine_displacement,
            'cylinder_count': cylinders,
            'cylinders': cylinders,  # Alias
            'accident': accident,
            'clean_title': clean_title,
            'ext_col': ext_color,
            'int_col': int_color
        }
        
        # Make prediction
        with st.spinner("🤖 Analyzing vehicle specifications..."):
            try:
                predicted_price = make_prediction(input_data)
                
                # Success message
                st.success("✅ Prediction Complete!")
                
                # Main prediction display
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='text-align: center; color: #1f77b4;'>Estimated Market Value</h2>
                    <h1 style='text-align: center; font-size: 4rem; color: #28a745;'>${predicted_price:,.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                st.subheader("📈 Additional Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    car_age = 2025 - model_year
                    st.metric("Car Age", f"{car_age} years")
                
                with col2:
                    depreciation_per_year = predicted_price / (car_age + 1) if car_age > 0 else predicted_price
                    st.metric("Annual Depreciation", f"${depreciation_per_year:,.0f}")
                
                with col3:
                    price_per_mile = predicted_price / mileage if mileage > 0 else 0
                    st.metric("Price per Mile", f"${price_per_mile:.2f}")
                
                with col4:
                    mileage_per_year_calc = mileage / (car_age + 1)
                    st.metric("Mileage/Year", f"{mileage_per_year_calc:,.0f}")
                
                # Price Range (Confidence Interval)
                st.divider()
                st.subheader("📊 Price Range Estimate (95% Confidence)")
                
                rmse = 66024  # Your model's RMSE
                margin = 1.96 * rmse
                lower_bound = max(0, predicted_price - margin)
                upper_bound = predicted_price + margin
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lower Bound", f"${lower_bound:,.0f}", delta=f"-${margin:,.0f}")
                
                with col2:
                    st.metric("Predicted Price", f"${predicted_price:,.0f}")
                
                with col3:
                    st.metric("Upper Bound", f"${upper_bound:,.0f}", delta=f"+${margin:,.0f}")
                
                st.info("💡 **Note:** Actual market prices may vary by ±$66,000 based on specific condition, location, and market factors.")
                
                # Vehicle Summary
                st.divider()
                st.subheader("🚗 Vehicle Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("**Basic Information:**")
                    st.write(f"• {model_year} {brand} {model_input}")
                    st.write(f"• {mileage:,} miles ({mileage_per_year_calc:,.0f} miles/year)")
                    st.write(f"• {car_age} years old")
                    st.write(f"• {fuel_type}")
                    st.write(f"• {transmission}")
                
                with summary_col2:
                    st.markdown("**Performance & Condition:**")
                    st.write(f"• {horsepower} HP / {cylinders} cylinders")
                    st.write(f"• {engine_displacement}L engine")
                    st.write(f"• Accident History: {accident}")
                    st.write(f"• Clean Title: {clean_title}")
                    st.write(f"• {ext_color} exterior / {int_color} interior")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.write("Please check your input values and try again.")
                st.write("If the problem persists, contact support.")
    
    else:
        st.info("👈 Fill in the vehicle details in the sidebar and click 'Predict Price' to get started!")

with tab2:
    st.header("📊 About the AI Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.write("""
        **Algorithm:** Random Forest Regressor  
        **Ensemble Size:** 300 decision trees  
        **Training Dataset:** 188,533 used cars  
        **Features:** 23 engineered features  
        **Validation:** External validation via Kaggle Competition
        """)
        
        st.subheader("Performance Metrics")
        st.write("""
        **Kaggle Private Score:** 66,024 RMSE  
        **Kaggle Public Score:** 74,801 RMSE  
        **Validation R²:** 0.120  
        **Test RMSE:** $66,024
        """)
    
    with col2:
        st.subheader("Top Features")
        st.write("""
        1. **Depreciation Indicator** - Mileage × Age
        2. **Mileage** - Total miles driven
        3. **Mileage per Year** - Usage intensity
        4. **Model** - Vehicle model encoding
        5. **Exterior Color** - Color preference
        6. **Interior Color** - Interior finish
        7. **Transmission** - Transmission type
        8. **Model Year** - Vehicle age
        9. **Car Age** - Years since manufacture
        10. **Engine Displacement** - Engine size
        """)
    
    st.divider()
    
    st.subheader("🎓 Academic Details")
    st.write("""
    **Student:** st20343580  
    **Module:** CIS 6005 - Computational Intelligence  
    **Institution:** Cardiff Metropolitan University  
    **Academic Year:** 2024-2025
    """)
    
    st.subheader("⚙️ Feature Engineering")
    st.write("""
    This model uses advanced feature engineering including:
    - **Temporal features:** Car age, mileage per year
    - **Luxury indicators:** Premium brand classification
    - **Performance metrics:** HP per cylinder, high-performance flags
    - **Depreciation modeling:** Non-linear interaction between age and usage
    - **Condition flags:** Accident history, title status
    """)

with tab3:
    st.header("❓ How to Use")
    
    st.write("""
    ### Step-by-Step Guide
    
    1. **Fill in Basic Information**
       - Select the vehicle brand from the dropdown
       - Enter the model name
       - Choose the model year
       - Enter total mileage
    
    2. **Specify Vehicle Specifications**
       - Select fuel type (Gasoline, Hybrid, Electric, etc.)
       - Choose transmission type
       - Enter horsepower and engine displacement
       - Select number of cylinders
    
    3. **Provide Condition Details**
       - Indicate if the vehicle has accident history
       - Specify if it has a clean title
       - Select exterior and interior colors
    
    4. **Get Prediction**
       - Click the "Predict Price" button
       - View the estimated market value
       - Check the confidence interval range
       - Review additional metrics
    
    ### Tips for Accurate Predictions
    
    ✅ **Provide Complete Information** - More accurate data leads to better predictions  
    ✅ **Use Actual Values** - Enter real specifications from the vehicle  
    ✅ **Check Similar Models** - Compare with market prices for similar vehicles  
    ✅ **Consider Condition** - Factor in actual vehicle condition beyond the inputs  
    ✅ **Location Matters** - Prices vary by region and market demand
    
    ### Understanding the Results
    
    - **Predicted Price:** The model's best estimate based on input features
    - **Price Range:** 95% confidence interval (±$66,000)
    - **Depreciation:** Estimated annual value decrease
    - **Price per Mile:** Cost per mile driven (lower is better)
    
    ### Limitations
    
    ⚠️ This is a predictive model trained on historical data. Actual market prices may vary based on:
    - Local market conditions
    - Specific vehicle condition details
    - Seasonal demand fluctuations
    - Recent maintenance and upgrades
    - Regional preferences
    
    ### Need Help?
    
    For questions or issues, please contact: **st20343580@cardiffmet.ac.uk**
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style='text-align: center; padding: 1rem; color: #666;'>
    <p><strong>Used Car Price Predictor</strong> | Powered by Machine Learning</p>
    <p>CIS 6005 - Computational Intelligence | Cardiff Metropolitan University</p>
    <p>© 2025 Student ID: st20343580</p>
</div>
""", unsafe_allow_html=True)