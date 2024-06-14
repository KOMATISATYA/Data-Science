import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from bokeh.plotting import figure
import graphviz
import plotly.figure_factory as ff
import plotly.express as px

chart_data1=pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
chart_data2=pd.DataFrame(
    {
        'col1':np.random.randn(20),
        'col2':np.random.randn(20),
        'col3':np.random.choice(['A','B','C'],20)
    }
)
# st.write(chart_data2)
#area_chart
st.write("Area chart:")
st.area_chart(chart_data1)
st.area_chart(chart_data2,x='col1',y='col2',color='col3')
st.area_chart(chart_data1,x='a',y=['b','c'],color=["#FF0000", "#0000FF"])

#bar_chart
st.write("Bar chart:")
st.bar_chart(chart_data1)
st.bar_chart(chart_data2,x='col1',y='col2',color='col3')
st.bar_chart(chart_data1,x='a',y=['b','c'],color=["#FF0000", "#0000FF"])

#line_chart
st.write("Line chart:")
st.line_chart(chart_data1)
st.line_chart(chart_data2,x='col1',y='col2',color='col3')
st.line_chart(chart_data1,x='a',y=['b','c'],color=["#FF0000", "#0000FF"])

df=pd.DataFrame(
    np.random.randn(1000,2)/[50,50]+[37.76,-122.4],
    columns=['lat','lon']
)
#map
st.write("Map:")
st.map(df)
st.map(df,size=20,color='#0044ff')

df = pd.DataFrame({
    "col1": np.random.randn(1000) / 50 + 37.76,
    "col2": np.random.randn(1000) / 50 + -122.4,
    "col3": np.random.randn(1000) * 100,
    "col4": np.random.rand(1000, 4).tolist(),
})

st.map(df,
    latitude='col1',
    longitude='col2',
    size='col3',
    color='col4')

#Scatter_chart

st.write("Scatter chart")
st.scatter_chart(chart_data1)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])
chart_data['col4'] = np.random.choice(['A','B','C'], 20)

st.scatter_chart(
    chart_data,
    x='col1',
    y='col2',
    color='col4',
    size='col3',
)

chart_data3=pd.DataFrame(np.random.randn(20,4),columns=["col1",'col2','col3','col4'])

st.scatter_chart(
    chart_data3,
    x='col1',
    y=['col2','col3'],
    size='col4',
    color=['#FF0000', '#0000FF']
)

#altair_chart
st.write("Altair chart")
c=(
    alt.Chart(chart_data1)
    .mark_circle()
    .encode(x='a',y='b',size='c',color='c',tooltip=['a','b','c'])
)

st.altair_chart(c,use_container_width=True)

#Bokeh_chart
st.write("Bokeh chart")

x=[1,2,3,4,5]
y=[6,7,2,4,5]

p=figure(
    title='simple line example',
    x_axis_label='x',
    y_axis_label='y'
)
p.line(x,y,legend_label='Trend',line_width=2)

st.bokeh_chart(p,use_container_width=True)

#Graphviz Chart

st.write("Graphviz chart")
graph=graphviz.Digraph()
graph.edge('run','intr')
graph.edge('intr','runbl')
graph.edge('runbl','run')
graph.edge('run','kernel')
graph.edge('kernel','zombie')
graph.edge('kernel','sleep')
graph.edge('kernel','runmem')
graph.edge('sleep','swap')
graph.edge('swap','runswap')
graph.edge('runswap','new')
graph.edge('runswap','runmem')
graph.edge('new', 'runmem')
graph.edge('sleep', 'runmem')

st.graphviz_chart(graph)

st.graphviz_chart('''
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
''')

#Plotly chart

st.write("Plotly chart")
x1=np.random.randn(200)-2
x2=np.random.randn(200)
x3=np.random.randn(200)+2

hist_data=[x1,x2,x3]

group_labels=['Group1','Group2','Group3']

fig=ff.create_distplot(
    hist_data,group_labels,bin_size=[.1,.25,.5]
    )
st.plotly_chart(fig,use_container_width=True)


df=px.data.iris()
fig=px.scatter(df,x='sepal_width',y='sepal_length')
event=st.plotly_chart(fig,key="iris",on_select='rerun')