# import packages
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# streamlit  background color
page_bg_color = '''
<style>
[data-testid="stAppViewContainer"]{
        background-color:#FFDAB9;
}
</style>'''

# streamlit button color
button_style = """
    <style>
        .stButton>button {
            background-color: #ffa089 ; 
            color: black; 
        }
        .stButton>button:hover {
            background-color: #ffddca; 
        }
    </style>    
"""
# streamlit settings
st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="auto")

st.markdown(page_bg_color, unsafe_allow_html=True)  # calling background color
st.markdown(button_style, unsafe_allow_html=True)  # calling button color

st.title("Industrial Copper Modeling")
# menu
selected = option_menu(menu_title=None, options=["HOME", "PREDICT SELLING PRICE", "PREDICT STATUS"],
                       icons=["house", "cash-coin", "trophy"],
                       default_index=0, orientation='horizontal',
                       styles={"container": {"background-color": "white", "size": "cover", "width": "100"},
                               "icon": {"color": "brown", "font-size": "20px"},

                               "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px",
                                            "--hover-color": "#ffe5b4"},
                               "nav-link-selected": {"background-color": "#E2838A"}})

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
        'status': [status],
        'item_type': [item_type],
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
        'item_type': [item_type],
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

# streamlit page
# home page
if selected == "HOME":
    col1, col2 = st.columns(2)
    with col1:
        st.image(r"D:\Copper\copper.jpg")
    with col2:
        st.write('''**The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions.
                 A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.**''')
        st.write('''**Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.**''')
    st.header(f"Regression: :red[ RandomForestRegressor]")
    st.write('* ML Regression model which predicts continuous variable :violet[**‘Selling_Price’**].')
    st.write(
        '- A Random forest regression model combines multiple decision trees to create a single model. Each tree in the forest builds from a different subset of the data and makes its own independent prediction. The final prediction for input is based on the average or weighted average of all the individual trees predictions.')

    st.header(f"Classification: :red[ RandomForestClassifier]")
    st.write('* ML Classification model which predicts Status: :green[**WON**] or :violet[**LOST**].')
    st.write(
        '- Random Forest Classification is an ensemble learning technique designed to enhance the accuracy and robustness of classification tasks. The algorithm builds a multitude of decision trees during training and outputs the class that is the mode of the classification classes.')

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
