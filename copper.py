# import packages
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
import base64

# streamlit settings
st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="auto")

 
st.markdown("""
    <style>
        .stButton>button {
            background-color: #0023F9 ; 
            color: black; 
        }
        .stButton>button:hover {
            background-color: #3400F9; 
        }
    </style>    
""", unsafe_allow_html=True) 


st.title("Industrial Copper Modeling")
# menu
selected = option_menu(menu_title=None, options=["PREDICT SELLING PRICE", "PREDICT STATUS"],
                       icons=["coin", "trophy"],
                       default_index=0, orientation='horizontal')


stat = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
coun = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
appli = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40.,
         25., 67., 79., 3., 87.5, 2., 5., 39., 69., 70., 65., 58., 68.]
product = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738, 332077137, 640405,
           1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026,
           164336407, 164337175, 1668701725, 1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219,
           1722207579, 929423819, 1665584320, 1665584662, 1665584642]

# load model and encoder
with open(r"D:\Copper\random_forest_regressor.pkl", 'rb') as file:
    loaded_reg_model = pickle.load(file)

with open(r"D:\Copper\item_type_label_encoder.pkl", 'rb') as f:
    type_loaded = pickle.load(f)

with open(r"D:\Copper\status_mapped.pkl", 'rb') as f:
    status_loaded = pickle.load(f)

with open(r"D:\Copper\random_forest_classifier.pkl", 'rb') as file1:
    loaded_class_model = pickle.load(file1)

# Ensure label encoder is fitted with all possible item types
all_item_types = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
type_loaded.fit(all_item_types)


# function selling price
def predict_price(quantity_tons, customer, country, status, item_type, application, thickness, product_ref, width, no_of_days):
    # Map status
    status_mapped = status_loaded.get(status)

    # Transform item type
    item_type_encoded = type_loaded.transform([item_type])[0]

    # Create input data DataFrame with consistent column names
    input_data = pd.DataFrame({
        'quantity_tons': [float(quantity_tons)], 
        'customer': [float(customer)], 
        'country': [float(country)], 
        'status': [status_mapped],
        'item_type': [item_type_encoded],
        'application': [float(application)], 
        'thickness': [float(thickness)], 
        'width': [float(width)], 
        'product_ref': [float(product_ref)], 
        'no_of_days': [float(no_of_days)] 
    })

    # Display input data for debugging
    st.write("Input data for prediction:")
    st.write(input_data)

    # Predict selling price
    prediction = loaded_reg_model.predict(input_data)

    # Display prediction
    st.write(f"#### :red[ Selling Price is $ {prediction[0]}]")

# function status
def predict_status(quantity_tons, customer, country, item_type, application, thickness, product_ref, width, selling_price, no_of_days):
    item_type_encoded = type_loaded.transform([item_type])[0]
    input_data_class = pd.DataFrame({
        'quantity_tons': [float(quantity_tons)], 
        'customer': [float(customer)], 
        'country': [float(country)], 
        'item_type': [item_type_encoded],
        'application': [float(application)], 
        'thickness': [float(thickness)], 
        'width': [float(width)], 
        'product_ref': [float(product_ref)], 
        'selling_price': [float(selling_price)], 
        'no_of_days': [float(no_of_days)] 
    })

    # Predict status
    prediction_status = loaded_class_model.predict(input_data_class)

    # Display prediction
    if prediction_status[0] == 1:
        st.write("#### :red[ Status is WON]")
    else:
        st.write("#### :red[ Status is LOST]")



# predict selling price page
if selected == "PREDICT SELLING PRICE":
    col1, col2 = st.columns(2)
    with col1:
        status = st.selectbox("Status", stat, key=1)
        item_type = st.selectbox("Item Type", item, key=2)
        country = st.selectbox("Country", sorted(coun), key=3)
        application = st.selectbox("Application", sorted(appli), key=4)
        product_ref = st.selectbox("Product Reference", product, key=5)

    with col2:
        quantity_tons = st.text_input("Enter Quantity Tons (Min:0 & Max:151.45)")
        thickness = st.text_input("Enter thickness (Min:0.18 & Max:6.45)")
        width = st.text_input("Enter width (Min:691, Max:1981)")
        customer = st.text_input("Customers (Min:30071590, Max:30405710)")
        no_of_days = st.text_input("Enter Delivery time taken(Min:0,Max:199)")

    if st.button("Predict price"):
        predict_price(quantity_tons, customer, country, status, item_type, application, thickness, product_ref, width, no_of_days)

# predict status page
if selected == "PREDICT STATUS":
    col3, col4 = st.columns(2)
    with col3:
        item_type = st.selectbox("Item Type", item)
        country = st.selectbox("Country", sorted(coun))
        application = st.selectbox("Application", sorted(appli))
        product_ref = st.selectbox("Product Reference", product)
        selling_price = st.text_input("Enter Selling price(Min:243,Max:1379)")

    with col4:
        quantity_tons = st.text_input("Enter Quantity Tons (Min:0 & Max:151.45)")
        thickness = st.text_input("Enter thickness (Min:0.18 & Max:6.45)")
        width = st.text_input("Enter width (Min:691, Max:1981)")
        customer = st.text_input("Customers (Min:30071590, Max:30405710)")
        no_of_days = st.text_input("Enter Delivery time taken(Min:0,Max:199)")

    if st.button("Predict Status"):
        predict_status(quantity_tons, customer, country, item_type, application, thickness, product_ref, width, selling_price, no_of_days)
