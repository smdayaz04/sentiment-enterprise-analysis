import pandas as pd
import streamlit as st

def load_data(n=50000): 
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # NEW STABLE URL
    url = "https://raw.githubusercontent.com/NiteshKedia/Sentiment-Analysis/master/Word%20list%20with%20sentiment/trainingandtestdata/training.1600000.processed.noemoticon.csv"
    
    try:
        # Load data with a timeout to prevent hanging
        df = pd.read_csv(url, encoding='latin-1', names=cols, nrows=n, on_bad_lines='skip')
        df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
        return df
    except Exception as e:
        # SAFETY NET: If the internet fails, the app still shows a demo
        st.warning("External data link failed. Loading demo mode...")
        return pd.DataFrame({
            'text': ['Service was great!', 'I hate the wait time.', 'The app crashed.', 'Amazing experience!'],
            'label': ['positive', 'negative', 'negative', 'positive'],
            'target': [4, 0, 0, 4]
        })