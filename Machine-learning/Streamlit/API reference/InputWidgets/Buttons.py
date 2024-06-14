import streamlit as st
import pandas as pd
import numpy as np
#Button
st.button("Reset",type="primary")
if st.button("Say hello"):
    st.write("Hello there..")
else:
    st.write("Good Bye..")

# Download button

# 1. Download a dataframe as a csv.

dataframe=pd.DataFrame(np.random.randn(20,3),columns=['A','B','C'])

def convert_df(df):
    return df.to_csv().encode("utf-8")

csv=convert_df(dataframe)

st.download_button(
    label='Download data as a csv',
    data=csv,
    file_name='dataframe.csv',
    mime='text/csv'
)

#Download string as a file
text_contents='''This is some text'''
st.download_button("Download some text",text_contents)

#Download a binary file
binary_contents=b"example content"
st.download_button("Download binary file", binary_contents)

#Download an image
with open("flower1.png",'rb') as file:
    btn=st.download_button(
        label="Download image",
        data=file,
        file_name="flower.png",
        mime="image/png"
    )

# Form submit button
with st.form(key='my_form'):
    
    name = st.text_input('Enter your name')
    age = st.number_input('Enter your age', min_value=0, max_value=120, step=1)
    agree = st.checkbox('I agree to the terms and conditions')
    # Add a submit button to the form
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.write(f'Name:{name}')
    st.write(f'age:{age}')
    st.write(f'Agreement: {"Yes" if agree else "No"}')

#Link button
st.link_button("Go to gallery","https://streamlit.io/gallery")

#Page link
st.page_link("http://www.google.com", label="Google", icon="🌎")
