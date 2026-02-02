import pandas as pd
import config

def load_data(n=None):
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    df = pd.read_csv(config.RAW_FILE, encoding='latin-1', names=cols, nrows=n)
    
    df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
    df['timestamp'] = pd.to_datetime(df['date'].str.replace(' PDT ', ' '), errors='coerce')
    
    return df