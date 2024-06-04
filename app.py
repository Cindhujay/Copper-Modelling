import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np

import pandas as pd
df = pd.read_csv(r"D:\Copper\copper_final.csv")


from sklearn.model_selection import train_test_split
#column = ['quantity tons','customer','country','application','thickness','width','product_ref','status','item type','no_of_days']
x = df.iloc[:, :10]
y = df['selling_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 7)


from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg.fit(x_train, y_train)
rfreg.score(x_test, y_test)


from sklearn.model_selection import train_test_split
x = df.iloc[:, :10]
y = df['status']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 , random_state = 7)

from imblearn.combine import SMOTEENN
cc = SMOTEENN(random_state=7)
x_train_smoteenn, y_train_smoteenn = cc.fit_resample(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
rf = RandomForestClassifier()
rf.fit(x_train_smoteenn,y_train_smoteenn)
f1_score(y_test,rf.predict(x_test))

customerid  = df['customer'].unique().tolist()
prod = df['product_ref'].unique().tolist()
country =df['country'].unique().tolist()
app = df['application'].unique().tolist()
items = df['item type'].unique().tolist()



st.set_page_config(page_title="Industrial_Copper_Modelling", page_icon=None, layout="wide")
st.header("Industrial_Copper_Modelling")
t1,t2 = st.tabs(["Copper price prediction","Status prediction"])

with t1:
    col1,col2,col3 = st.columns(3)
    with col1:
        customer = st.selectbox('Enter your Customer ID',customerid)
        thickness = st.text_input('Enter thickness of copper')
        product_ref = st.selectbox('Select the product reference',prod)

    with col2:
        country = st.selectbox('Select the country code',(28.0,38.0,78.0,27.0,30.0,32.0,25.0,77.0,79.0,39.0,40.0,26.0,84.0,80.0,113.0,89.0))
        quantity_tons = st.text_input('Enter the quantity in tons')
        status = st.selectbox('Select the status code',('Won','Lost'))
        sta = {"Won":1.0,"Lost":0.0}
        nday = st.text_input('Enter the expected number of days for delievery')

    with col3:
        application = st.selectbox('Enter your application code',app)
        width = st.text_input('Enter width of copper')
        item_type = st.selectbox('Select the item type',('W','S','Others','PL','WI','IPL'))
        ite={'W':4.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'WI':5.0, 'IPL':0.0}
    c1,c2,c3 = st.columns([3,1,3])
    with c2:
        submit = submit = st.button("Predict")
        if submit:
            data = np.array([[quantity_tons,customer,country,sta[status],ite[item_type],application,thickness,width,product_ref,nday]])
            pred_r = rfreg.predict(data)
            st.write(f"The selling Price is {pred_r}")
with t2:
    col1,col2,col3 = st.columns(3)
    with col1:
        customer_ = st.selectbox(' Enter your Customer ID',customerid)
        thickness_ = st.text_input(' Enter thickness of copper')
        product_ref_ = st.selectbox(' Select the product reference',prod)

    with col2:
        country_ = st.selectbox(' Select the country code',(28.0,38.0,78.0,27.0,30.0,32.0,25.0,77.0,79.0,39.0,40.0,26.0,84.0,80.0,113.0,89.0))
        quantity_tons_ = st.text_input(' Enter the quantity in tons')
        selling_price_ = st.text_input(' Enter the selling price of copper')
        nday_ = st.text_input(' Enter the expected number of days for delievery')

    with col3:
        application_ = st.selectbox(' Enter your application code',app)
        width_ = st.text_input(' Enter width of copper')
        item_type_ = st.selectbox(' Select the item type',('W','S','Others','PL','WI','IPL'))
        ite={'W':4.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'WI':5.0, 'IPL':0.0}
    c1,c2,c3 = st.columns([3,1,3])
    with c2:
        submit_ = st.button(" Predict")
        if submit_:
            data1 = np.array([[quantity_tons_,customer_,country_,ite[item_type_],application_,thickness_,width_,product_ref_,selling_price_,nday_]])
            pred_c = rf.predict(data1)
            status_message = "The status is Won" if pred_c[0] == 1 else "The status is Lost"
            st.write(status_message)
