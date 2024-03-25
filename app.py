import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page title and favicon
st.set_page_config(page_title="Laptop Price Predictor", page_icon=":computer:")

# Custom CSS styles for improved layout
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 24px;
        margin-bottom: 10px;
    }
    .section-header-custom {
        color: #FF5733; /* Change header color */
    }
    .prediction {
        font-size: 36px; /* Increase font size */
        text-align: center;
        margin-top: 30px;
        color: #2E86C1; /* Change prediction result color */
        border: 2px solid #2E86C1; /* Add border */
        border-radius: 10px; /* Add border radius */
        padding: 10px; /* Add padding */
    }
    .input-column {
        float: left;
        width: 33.33%;
        padding: 10px;
    }
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Laptop Price Predictor")

# Brand, type, MS Office, Backlit Keyboard, Warranty, Weight
with st.container():
    st.header("Basic Information")
    company = st.selectbox('Brand', df['brand'].unique())
    type = st.selectbox('Type', df['type'].unique())
    msoffice = st.radio('MS Office', ['Yes', 'No'])
    backlitkb = st.radio('Backlit Keyboard', ['Yes', 'No'])
    warranty = st.selectbox('Warranty(in years)', ['1', '2', '3'])
    weight = st.number_input('Weight of the Laptop (in kg)')

# OS, resolution, Touchscreen, Refresh Rate
with st.container():
    st.header("Display & Operating System")
    os = st.selectbox('OS', df['OS'].unique())
    resolution = st.selectbox('Screen Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                               '2560x1600', '2560x1440', '2304x1440'])
    touchscreen = st.radio('Touchscreen', ['Yes', 'No'])
    screen_size = st.number_input('Screen Size (in inches)', min_value=10.0, max_value=30.0, value=15.6)
    rr = st.selectbox('Refresh Rate(in Hz)', [60, 90, 120, 144, 165, 240, 360])

# RAM, HDD, SSD, CPU, GPU
with st.container():
    st.header("Hardware")
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024, 2048])
    cpu = st.selectbox('CPU', df['Processor'].unique())
    gpu = st.selectbox('GPU', df['GPU'].unique())

if st.button('Predict Price', key='predict_btn'):
    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare query data
    query = pd.DataFrame({
        'brand': [company],  # Add the 'brand' column
        'type': [type],
        'msoffice': [1 if msoffice == 'Yes' else 0],
        'RAM': [ram],
        'SSDCap': [ssd],
        'HDDCap': [hdd],
        'GPU': [gpu],
        'OS': [os],
        'Touchscreen': [1 if touchscreen == 'Yes' else 0],
        'refreshrate': [rr],
        'Weight': [weight],
        'BacklitKB': [1 if backlitkb == 'Yes' else 0],
        'warranty': [int(warranty)],
        'PPI': [ppi],
        'Processor': [cpu]
    })

    # Make prediction
    predicted_price = np.exp(pipe.predict(query)[0])
    st.subheader("Prediction")
    st.markdown("<div class='prediction'>The predicted price of this configuration is ₹" + str(int(predicted_price)) + "</div>", unsafe_allow_html=True)
