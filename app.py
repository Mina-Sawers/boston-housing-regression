import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor

# st.set_page_config(page_title='Boston Home Price Predictor', page_icon='🏠', layout='wide')

st.title('Boston Home Price Predictor')

@st.cache_resource
def load_model():
    return joblib.load('model/decision_tree_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/housing.csv')

model = load_model()
data = load_data()

st.sidebar.header('Property Features')

rooms = st.sidebar.slider('Total number of rooms',min_value=1,max_value=10,value=5)
poverty = st.sidebar.slider('Poverty rate (%)',min_value=1,max_value=100,value=10)
st_ratio = st.sidebar.slider('Student-teacher ratio',min_value=1,max_value=30,value=15)

if st.button("Predict Home Price"):
    input_data = np.array([[rooms,poverty,st_ratio]])
    prediction = model.predict(input_data)[0]
    st.success("Prediction calculated successfully!")
    st.metric(label='Estimated Value',value=f"${prediction:,.2f}")


st.markdown('---')

st.subheader('Market Analysis')

col1,col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        data, x='RM',y='MEDV',
        title='Rooms vs. Home Price',
        labels={'RM':'Rooms','MEDV':'Price ($)'},
        color_discrete_sequence=['#636EFA']
    )
    fig1.add_vline(x=rooms,line_dash='dash',line_color='red',annotation_text='Your Input')
    st.plotly_chart(fig1,use_container_width=True)

with col2:
    fig2 = px.scatter(
        data, x='LSTAT',y='MEDV',
        title='Poverty Rate vs. Home Price',
        labels={'LSTAT':'Poverty Rate (%)','MEDV':'Price ($)'},
        color_discrete_sequence=['#a200ff']
    )
    fig2.add_vline(x=poverty,line_width=2,line_dash='dash',line_color='red',annotation_text="Your Input")
    st.plotly_chart(fig2,use_container_width=True)

# col3 = st.columns(1)

# with col3:
fig3 = px.scatter(
    data,x='PTRATIO',y='MEDV',
    title = "Student-Teacher Ratio vs. Home Price",
    labels={'PTRATIO':'Students To Teacher Ratio','MEDV':'Price ($)'},
    color_discrete_sequence=["#0088ff"]
)
fig3.add_vline(x=st_ratio,line_width=2,line_dash='dash',line_color='red',annotation_text='Your Input')
st.plotly_chart(fig3,use_container_width=True)

st.markdown("---")

# 6. Signature and Project Details
st.markdown("""
### About this Project
This interactive dashboard predicts real estate prices using the classic **Boston Housing Dataset**. It is powered by a machine learning **Decision Tree Regressor** trained on three key market drivers: the number of rooms (RM), neighborhood poverty rate (LSTAT), and local student-teacher ratio (PTRATIO). 

To prevent the algorithm from overfitting to the training data, the model was optimized with a **Maximum Depth of 4**, striking the ideal balance between complexity and predictive generalization.

👨‍💻 **Created by Mina Sawiris**  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/mina-sawiris) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/mina-sawers/)
""")