import pandas as pd
import config

def load_data(n=None):
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # This URL is the raw version of the Kaggle Sentiment140 data
    url = "https://raw.githubusercontent.com/kaz-Anova/Sentiment140/master/training.1600000.processed.noemoticon.csv"
    
    df = pd.read_csv(url, encoding='latin-1', names=cols, nrows=n)
    df['label'] = df['target'].replace({0: 'negative', 4: 'positive'})
    return df