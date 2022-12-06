import streamlit as st
import re
import pandas as pd
import emoji
from arabert.preprocess import ArabertPreprocessor

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


def load_arabert():
    model_name = "aubmindlab/bert-base-arabertv2"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    return arabert_prep

@st.cache
def preprocess_text(text):
  text = re.sub("[a-zA-Z]", " ", text) # remove english letters
  text = re.sub(r'[ء-ي]+@', '', text) # remove profile tags
  text = re.sub(r'[ء-ي]+#', '', text) # remove profile tags
  arabert_prep = load_arabert()
  text = arabert_prep.preprocess(text)
  return text


def get_prediction(text, model):
    result = model(text).cats
    label = sorted(result, key=result.get, reverse=True)[0]
    score = result[label]

    return label, score

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache
def classify_from_df(df, column):
    predicted_classes_off = []
    scores_off = []
    predicted_classes_misg = []
    scores_misg = []
    for t in list(df[column]):
        text_cleaned = preprocess_text(t)
        # offensive
        label, score = get_prediction(text_cleaned, st.session_state['nlp_off'])
        label = label.replace('_', ' ')
        predicted_classes_off.append(label)
        scores_off.append(score)
        # misogyny
        label, score = get_prediction(text_cleaned, st.session_state['nlp_misg'])
        label_misg = 'not Misogyny' if label == 'none' else 'Misogyny'
        predicted_classes_misg.append(label_misg)
        scores_misg.append(score)
    
    
    df_output = pd.DataFrame(list(zip(list(df[column]), predicted_classes_off, scores_off, predicted_classes_misg, scores_misg)), columns=['Text', 'Offensiveness (predicted)', 'Score (Offensiveness)', 'Misogyny  (predicted)', 'Score (Misogyny)'])
    return df_output