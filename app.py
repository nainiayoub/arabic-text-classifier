import streamlit as st
import spacy
from functions import clean_text, get_prediction, convert_df, classify_from_df
import pandas as pd


html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """


with st.sidebar:
    input_options = st.selectbox("Input option", ['Text', 'CSV File'])

st.title("VAW in Libya")
st.write("Coarse-grained text classification of Arabic text into offensive / not offenisve.")
st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)

# loading model
if 'nlp' not in st.session_state:
    nlp = spacy.load('./model-last')
    st.session_state['nlp'] = nlp


if input_options == 'Text':
    text = st.text_area("Enter Arabic text")
    if text:
        text_cleaned = clean_text(text)
        label, score = get_prediction(text_cleaned)
        col1, col2 = st.columns(2)
        with col1:
            st.info(label)
        with col2:
            st.warning("Score: "+str(score))
else:
    uploaded_file = st.file_uploader("Enter test data (csv)", type=['csv'])
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        with st.expander("Dataframe"):
            st.table(dataframe)

        with st.sidebar:
            columns = dataframe.columns.tolist()
            text_column = st.selectbox('Which is the text column?', ['None']+columns)

        if text_column in columns:
            df_output = classify_from_df(dataframe, text_column)
            with st.expander("Output Dataframe"):
                st.table(df_output)

            csv = convert_df(df_output)

            st.download_button(
                label="Download output dataframe as CSV",
                data=csv,
                file_name='classified_data.csv',
                mime='text/csv',
            )





