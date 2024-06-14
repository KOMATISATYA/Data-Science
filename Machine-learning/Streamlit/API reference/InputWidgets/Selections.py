import streamlit as st

# Check box 
agree = st.checkbox("I agree")
if agree:
    st.write("Great!!")

# Color picker
color=st.color_picker("Pick a color","#00f900")
st.write("The currect color is",color)

# Multi color
options=st.multiselect("What is your favourite colors",
                       ["Green","Yellow","Red",'Blue'],
                       ['Yellow',"Red"])
st.write("You selected:", options)