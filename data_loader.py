import pandas as pd
import config

def load_data(n=None):
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Use the URL from config instead of a local file path
    df = pd.read_csv(config.DATA_URL, encoding='latin-1', names=cols, nrows=n)
    
    df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
    return df