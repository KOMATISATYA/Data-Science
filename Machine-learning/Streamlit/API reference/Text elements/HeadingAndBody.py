import streamlit as st


#title
st.title("This is title")
st.title("_Streamlit_ is :blue[cool] :sunglasses:")

#header
#divider (bool or “blue”, “green”, “orange”, “red”, “violet”, “gray”/"grey", or “rainbow”)

st.header("This is a header with divider",divider='rainbow')
st.header('streamlit is :orange[good] :smile:')

#subheader
st.subheader("This is a header with divider",divider='rainbow')

#markdown
st.markdown("*Streamlit* is **really** ***cool***.")
st.markdown('''
            :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
            :gray[pretty] :rainbow[colors] and :red-background[highlight] text.
            ''')
multi = '''If you end a line with two spaces,
a soft return is used for the next line.

Two (or more) newline characters in a row will result in a hard return.
'''
st.markdown(multi)

md=st.text_area('Type in your markdown string (without outer quotes)',
                "Happy Streamlit-ing! :balloon:")
st.code(f"""
        import streamlit as st
        
        st.markdown('''{md}''')
        """)
st.markdown(md)