import pandas as pd
import streamlit as st

def load_data(n=100000): 
    # These must match exactly what app.py expects
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    url = "https://raw.githubusercontent.com/NiteshKedia/Sentiment-Analysis/master/Word%20list%20with%20sentiment/trainingandtestdata/training.1600000.processed.noemoticon.csv"
    
    try:
        # The 'names=cols' part is what prevents the KeyError
        df = pd.read_csv(url, encoding='latin-1', names=cols, nrows=n, on_bad_lines='skip')
        df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
        return df
    except Exception as e:
        st.warning("Data link failed. Loading demo mode...")
        return pd.DataFrame(columns=cols) # Return empty DF with correct columns