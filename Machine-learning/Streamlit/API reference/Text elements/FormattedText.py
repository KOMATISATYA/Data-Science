import streamlit as st

st.title("Titlee...")

#caption
st.caption("This is the string that explains something above")
st.caption("A caption with _italics_ :blue[colors] and emojis :sunglasses:")

#code
code = '''def hello():
      print("Hello") '''
st.code(code,language='python')

#divider
st.divider()

st.write("This is some text.")
st.slider("This is slider",0,100,(25,75))
st.divider()
st.write("This is another text.")
st.divider()

#echo function is a context manager that captures the code inside its block and displays it on the Streamlit app along with its output.
#code location is a parameter it has values above and below
with st.echo():
    st.write('This code will be printed')

def get_user_name():
    return 'John'

with st.echo():
    # Everything inside this block will be both printed to the screen
    # and executed.

    def get_punctuation():
        return '!!!'

    greeting = "Hi there, "
    value = get_user_name()
    punctuation = get_punctuation()

    st.write(greeting, value, punctuation)

st.write('Done!')

#latex
st.latex(r'''
        a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} = 
        \sum_{k=0}^{n-1} ar^k=
        a \left(\frac{1-r^{n}}{1-r}\right)  
         ''')

#text
st.text('This is some text.')