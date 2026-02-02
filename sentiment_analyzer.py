from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config

sid = SentimentIntensityAnalyzer()

def analyze_sentiment(df):
    scores = []
    labels = []
    
    for text in df['clean_text']:
        s = sid.polarity_scores(str(text))['compound']
        scores.append(s)
        
        if s >= config.LIMITS['pos']:
            labels.append('positive')
        elif s <= config.LIMITS['neg']:
            labels.append('negative')
        else:
            labels.append('neutral')
            
    df['vader_score'] = scores
    df['vader_label'] = labels
    return df