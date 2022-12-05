import streamlit as st
import re
import pandas as pd
import emoji

@st.cache
def clean_text(text):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|?�!”…“#•–ـ''' # define arabic punctuations
    text = re.sub(r'@\S+', '', text) # remove profile tags
    text = re.sub("[a-zA-Z]", " ", text) # remove english letters
    text = re.sub('\n', ' ', text) # remove \n from text
    text = re.sub(r'\d+', '', text) #remove number
    text = re.sub(r'http\S+', '', text) # remove links
    text = re.sub(r'#\S+', '', text) # remove hashtags
    text = emoji.replace_emoji(text, '') # remove emojies
    # text = text.translate(str.maketrans('','', arabic_punctuations)) # remove punctuation
    text = re.sub(' +', ' ',text) # remove extra space
    text = text.strip() #remove whitespaces
    text = text.replace('@', '')


    return text

@st.cache
def get_prediction(text):
    result = st.session_state['nlp'](text).cats
    label = sorted(result, key=result.get, reverse=True)[0]
    score = result[label]

    return label, score

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache
def classify_from_df(df, column):
    predicted_classes = []
    scores = []
    for t in list(df[column]):
        text_cleaned = clean_text(t)
        label, score = get_prediction(text_cleaned)
        predicted_classes.append(label)
        scores.append(score)

    df_output = pd.DataFrame(list(zip(list(df[column]), predicted_classes, scores)), columns=['Text', 'Class', 'Score'])
    return df_output