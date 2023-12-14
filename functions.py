import streamlit as st
import re
import pandas as pd
import emoji
import os
import spacy
from arabert.preprocess import ArabertPreprocessor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, classification_report

st.cache_data
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

@st.cache(allow_output_mutation=True)
def load_arabert():
    model_name = "aubmindlab/bert-base-arabertv2"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    return arabert_prep

st.cache_data
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

st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

st.cache_data
def compute_score(target, predicted, score):
    if score == 'accuracy':
        return accuracy_score(target, predicted)
    elif score == 'recall':
        return recall_score(target, predicted, pos_label='not Misogyny')
    elif score == 'precision':
        return precision_score(target, predicted, pos_label='not Misogyny')
    elif score == 'f1_score':
        return f1_score(target, predicted, pos_label='not Misogyny')

st.cache_data
def classify_from_df(df, column):
    predicted_classes_off = []
    scores_off = []
    letmi = []
    armis = []
    letmi_armis = []
    fb_votes = []
    scores_misg = []
    path = './models/vaw_misogyny_models/'
    misg = {}
    misg_paths = []
    models = []
    # get misogyny models
    for i in os.listdir(path):
        models.append(i)
        misg_paths.append(path+i)
        
    # load misogyny models
    loaded_misg_models = []
    for i in misg_paths:
        loaded_misg_models.append(spacy.load(i))
        
    # classify with models
    predictions_misg = []
    
    for t in list(df[column]):
        text_cleaned = preprocess_text(t)
        # offensive
        label, score = get_prediction(text_cleaned, st.session_state['nlp_off'])
        label = label.replace('_', ' ')
        predicted_classes_off.append(label)
        scores_off.append(score)
        # misogyny
        for i in range(len(loaded_misg_models)):
            label_misg, score_misg = get_prediction(text_cleaned, loaded_misg_models[i])
            label_misg = 'not Misogyny' if label_misg == 'none' else 'Misogyny'
            if models[i] not in misg.keys():
                misg[models[i]] = []

            misg[models[i]].append(label_misg)

        
    
    
    df_output = pd.DataFrame(list(zip(list(df[column]), predicted_classes_off, scores_off, misg['ArMIS'], misg['LetMI'], misg['LetMI_ArMIS'], misg['Model_fb_data_votes'])), columns=['Text', 'Offensiveness (predicted)', 'Score (Offensiveness)', 'Misogyny (ArMIS)', 'Misogyny (LetMI)', 'Misogyny (LetMI_ArMIS)', 'Misogyny (Model_fb_data_votes)'])
    return df_output
