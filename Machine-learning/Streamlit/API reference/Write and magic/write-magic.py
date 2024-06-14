import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


st.write("Helloo")
st.write("*World*")
st.write(":smile:")
st.write(1234)

data=pd.DataFrame({
    'first column':[1,2,3,4],
    'second column':[10,20,30,40]
})

st.write(data)

st.write("data frame is",data)


df = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c'])

c = alt.Chart(df).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

st.write(c)

