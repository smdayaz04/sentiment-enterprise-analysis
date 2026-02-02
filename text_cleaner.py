import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPS = set(stopwords.words('english'))

def clean_tweet(txt):
    if not isinstance(txt, str): return ""
    
    txt = txt.lower()
    txt = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', txt)
    txt = re.sub(r'[^a-z0-9\s]', '', txt)
    
    words = [w for w in txt.split() if w not in STOPS]
    return " ".join(words).strip()

def process_batch(df):
    df['clean_text'] = df['text'].apply(clean_tweet)
    return df