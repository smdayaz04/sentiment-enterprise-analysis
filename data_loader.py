import pandas as pd
import streamlit as st

def load_data(n=100000): # We load 100k rows for speed, but you can set to None
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Using a compressed ZIP version of the same dataset to avoid HTTP Errors
    url = "https://github.com/kaz-Anova/Sentiment140/raw/master/training.1600000.processed.noemoticon.csv.zip"
    
    try:
        # Pandas can read ZIP files directly if you specify the compression
        df = pd.read_csv(url, encoding='latin-1', names=cols, nrows=n, compression='zip')
        df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()