import streamlit as st
import spacy
from functions import clean_text, get_prediction, convert_df, classify_from_df, load_arabert, preprocess_text
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


st.set_page_config(
    page_title="Arabic Text Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """


with st.sidebar:
    input_options = st.selectbox("Input option", ['Text', 'CSV File'])


st.title("VAW in Libya")
st.write("Coarse-grained text classification of Arabic text for offensiveness and misogyny.")
st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)

# loading model
if 'nlp_off' not in st.session_state:
    nlp_off = spacy.load('./models/model-last-offensive-v2')
    nlp_misg = spacy.load('./models/model-last-misogyny')
    st.session_state['nlp_off'] = nlp_off
    st.session_state['nlp_misg'] = nlp_misg


if input_options == 'Text':
    text = st.text_area("Enter Arabic text", "ناقصة عقل ودين أنت يا صاحبة المنشور")
    if text:
        text_cleaned_off = clean_text(text)
        label_off, score_off = get_prediction(text_cleaned_off, st.session_state['nlp_off'])
        text_cleaned_misg = preprocess_text(text)
        label_misg, score_misg = get_prediction(text_cleaned_misg, st.session_state['nlp_misg'])
        label_misg = 'not_misogynistic' if label_misg == 'none' else 'misogynistic'
        col1, col2 = st.columns(2)
        with col1:
            st.info(label_off)
            st.info(label_misg)
        with col2:
            st.warning("Score: "+str(score_off))
            st.warning("Score: "+str(score_misg))
        

        
        
else:
    uploaded_file = st.file_uploader("Enter test data (csv)", type=['csv'])
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        with st.expander("Dataframe"):
            st.dataframe(dataframe)

        with st.sidebar:
            columns = dataframe.columns.tolist()
            text_column = st.selectbox('Which is the text column?', ['None']+columns)

        if text_column in columns:
            df_output = classify_from_df(dataframe, text_column)
            with st.expander("Output Dataframe (predictions)"):
                st.dataframe(df_output)

                csv = convert_df(df_output)

                st.download_button(
                    label="Download output dataframe as CSV",
                    data=csv,
                    file_name='classified_data.csv',
                    mime='text/csv',
                )

            with st.sidebar:
                columns_tar = dataframe.columns.tolist()
                columns_pre = df_output.columns.tolist()
                st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
                st.markdown("""
                ## Evaluating Offensiveness
                """)
                target_off = st.selectbox('Select target class for offensiveness', ['None']+columns_tar)
                predicted_off = st.selectbox('Select predicted class for offensiveness', ['None']+columns_pre)
                st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
                st.markdown("""
                ## Evaluating Misogyny
                """)
                target_misg = st.selectbox('Select target class for misogyny', ['None']+columns_tar)
                predicted_misg = st.selectbox('Select predicted class for misogyny', ['None']+columns_pre)

            if target_off != 'None' and predicted_off != 'None':
                target = list(dataframe[target_off])
                predicted = list(df_output[predicted_off])
                st.markdown("""
                    #### Evaluating Offensiveness
                """)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Accuracy", value=str('{:.4f}'.format(accuracy_score(target, predicted))))
                with col2:
                    st.metric(label="Precision", value=str('{:.4f}'.format(precision_score(target, predicted, pos_label='offensive'))))
                with col3:    
                    st.metric(label="Recall", value=str('{:.4f}'.format(recall_score(target, predicted, pos_label='offensive'))))
                with col4:
                    st.metric(label="F1-score", value=str('{:.4f}'.format(f1_score(target, predicted, pos_label='offensive'))))

            if target_misg != 'None' and predicted_misg != 'None':
                target = list(dataframe[target_misg])
                predicted = list(df_output[predicted_misg])
                st.markdown("""
                    #### Evaluating Misogyny
                """)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Accuracy", value=str('{:.4f}'.format(accuracy_score(target, predicted))))
                with col2:
                    st.metric(label="Precision", value=str('{:.4f}'.format(precision_score(target, predicted, pos_label='Misogyny'))))
                with col3:    
                    st.metric(label="Recall", value=str('{:.4f}'.format(recall_score(target, predicted, pos_label='Misogyny'))))
                with col4:
                    st.metric(label="F1-score", value=str('{:.4f}'.format(f1_score(target, predicted, pos_label='Misogyny'))))
                    
               






