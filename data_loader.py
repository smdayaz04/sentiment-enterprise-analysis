import pandas as pd
import streamlit as st

def load_data(n=50000): 
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    url = "https://raw.githubusercontent.com/skandermoalla/multi-modal-emotion-recognition/master/data/sentiment140/training.1600000.processed.noemoticon.csv"

    try:
        df = pd.read_csv(url, encoding='latin-1', names=cols, nrows=n, on_bad_lines='skip')
        
        if 'date' not in df.columns:
            df['date'] = "Mon May 11 03:17:40 PDT 2009"
            
        df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
        return df
    except Exception as e:
        return pd.DataFrame({
            'target': [0, 4], 'date': ['Mon May 11 2009', 'Mon May 11 2009'],
            'text': ['Error loading remote data', 'System in fallback mode'],
            'label': ['negative', 'positive']
        })