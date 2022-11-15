import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
st.markdown('''
# **Exploaratory Data Analysis WebApp**
This App is develpoed by Codanics youtube channel "EDA APP" 
 ''')

with st.sidebar.header("Upload Your Dataset (.csv)"):
    uploaded_file= st.sidebar.file_uploader("upload your file", type=["csv"])
    df= sns.load_dataset('titanic')
    st.sidebar.markdown(f'[Example Dataset](https://docs.streamlit.io/streamlit-cloud/troubleshooting)')
    
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv= pd.read_csv(uploaded_file)
        return csv
    df= load_csv()
    pr= ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('Profiling Report with Pandas')
    st_profile_report(pr)
else:
    st.info("Please Upload a file")
    if st.button("Press to use Example data"):
        @st.cache
        def example_data():
            a= pd.DataFrame(np.random.rand(100,5), columns=['a','b','c','d','e'])
            return a
        df= example_data()
        pr= ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('Profiling Report with Pandas')
        st_profile_report(pr)


