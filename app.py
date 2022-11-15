# Name: Ovaimer Ali
# Date: 13-11-2022
# email: ovaimerghanni545@gmail.com
# Assignment: Streamlit WebApp

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# make container
header= st.container()
data_sets=st.container()
features=st.container()
model_training=st.container()

with header:
    st.title("kashti ka Dataset")
    st.text("In this Project we will work on kashti dataset")

with data_sets:
    st.header("haww! Kashti doob gi")
    st.text("we will work with titanic dataset")
    df= sns.load_dataset("titanic")
    df=df.dropna()
    st.write(df.head(10))
    
    # Bar chart
    st.subheader("Gender k hisab sy log")
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.subheader("People in Different Class")
    st.bar_chart(df['class'].value_counts())

    # Bar chart of age
    st.bar_chart(df['age'].sample(20))

with features:
    st.header("This app has some features")
    st.text("In this Project we will work on kashti dataset")
    st.markdown("1. **Feature 1:** This is feature one of this code")
    st.markdown("2. **Feature 2:** This is feature two of this code")
    st.markdown("3. **Feature 3:** This is feature three of this code")
    
with model_training:
    st.header("Kashti walon ka kya bna ML training model")
    st.text("اس میں ھم مشین لرننگ کا ماڈل استعمال کریں گے")
    # Lets make columns
    input, display = st.columns(2)
    max_depth=input.slider("How many people do you know?", min_value=10,max_value=100,value=20,step=5)

# n_estimator
n_estimator=input.selectbox("How many Trees are there in RF",options=[100,200,300,'No limit'])

#adding list of features
input.write(df.columns)
# input features from user
input_features=input.text_input("Which Feature we have to use?")


# Machine Learning  
model= RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimator)

# if condition for no limit
if n_estimator=='No limit':
    model=RandomForestRegressor(max_depth=max_depth)
else:
    model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimator)
# define X and y
X=df[[input_features]]
y=df[['fare']]

# fit the model
model.fit(X, y)
pred=model.predict(y)

# Display Metrices
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R squared score of the model is: ")
display.write(r2_score(y,pred))








