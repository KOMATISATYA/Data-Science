import streamlit as st
import pandas as pd
import numpy as np
import random

#dataframe
df=pd.DataFrame(np.random.randn(50,20),columns=("col %d" % i for i in range(20)))
st.dataframe(df) # same as st.write(df)
df=pd.DataFrame(np.random.randn(10,20),columns=("col %d" % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

df = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)
st.dataframe(
    df,
    column_config={
        "name": "App name",
        "stars": st.column_config.NumberColumn(
            "Github Stars",
            help="Number of stars on GitHub",
            format="%d ⭐",
        ),
        "url": st.column_config.LinkColumn("App URL"),
        "views_history": st.column_config.LineChartColumn(
            "Views (past 30 days)", y_min=0, y_max=5000
        ),
    },
    hide_index=True,
)

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(
        np.random.randn(12, 5), columns=["a", "b", "c", "d", "e"]
    )

st.dataframe(st.session_state.df)

event = st.dataframe(
    st.session_state.df,
    key="data",
    on_select="rerun",
    selection_mode=["multi-row", "multi-column"],
)

event.selection

df1 = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
my_table = st.table(df1)
df2 = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
my_table.add_rows(df2)
my_chart = st.line_chart(df1)
my_chart.add_rows(df2)