import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config

class CrisisDetector:
    
    def __init__(self):
        self.alerts = []
    
    def calculate_sentiment_metrics(self, df, time_column='date'):
        if 'predicted_sentiment' not in df.columns:
            print("No sentiment predictions found")
            return None
        
        metrics = {
            'total_count': len(df),
            'positive_count': len(df[df['predicted_sentiment'] == 'positive']),
            'neutral_count': len(df[df['predicted_sentiment'] == 'neutral']),
            'negative_count': len(df[df['predicted_sentiment'] == 'negative']),
            'avg_compound': df['vader_compound'].mean(),
            'median_compound': df['vader_compound'].median()
        }
        
        metrics['positive_pct'] = (metrics['positive_count'] / metrics['total_count']) * 100
        metrics['neutral_pct'] = (metrics['neutral_count'] / metrics['total_count']) * 100
        metrics['negative_pct'] = (metrics['negative_count'] / metrics['total_count']) * 100
        
        return metrics
    
    def detect_negative_spike(self, metrics):
        negative_threshold = config.CRISIS_CONFIG['negative_threshold']
        
        if metrics['negative_pct'] > negative_threshold:
            alert = {
                'type': 'HIGH_NEGATIVITY',
                'severity': 'HIGH',
                'message': f"Negative sentiment at {metrics['negative_pct']:.1f}% (threshold: {negative_threshold}%)",
                'value': metrics['negative_pct']
            }
            return alert
        return None
    
    def detect_sentiment_drop(self, df, window_hours=24):
        if 'date' not in df.columns or df['date'].isna().all():
            return None
        
        df_sorted = df.sort_values('date')
        
        if len(df_sorted) < 100:
            return None
        
        split_point = int(len(df_sorted) * 0.5)
        early_data = df_sorted.iloc[:split_point]
        recent_data = df_sorted.iloc[split_point:]
        
        early_negative_pct = (len(early_data[early_data['predicted_sentiment'] == 'negative']) / len(early_data)) * 100
        recent_negative_pct = (len(recent_data[recent_data['predicted_sentiment'] == 'negative']) / len(recent_data)) * 100
        
        drop = recent_negative_pct - early_negative_pct
        spike_threshold = config.CRISIS_CONFIG['spike_threshold']
        
        if drop > spike_threshold:
            alert = {
                'type': 'SENTIMENT_DROP',
                'severity': 'CRITICAL',
                'message': f"Sentiment dropped by {drop:.1f}% (threshold: {spike_threshold}%)",
                'value': drop
            }
            return alert
        
        return None
    
    def analyze_for_crisis(self, df):
        print("\nRunning Crisis Detection Analysis...")
        
        metrics = self.calculate_sentiment_metrics(df)
        
        if metrics is None:
            return None
        
        print(f"\nCurrent Sentiment Status:")
        print(f"  Positive: {metrics['positive_pct']:.1f}%")
        print(f"  Neutral: {metrics['neutral_pct']:.1f}%")
        print(f"  Negative: {metrics['negative_pct']:.1f}%")
        
        self.alerts = []
        
        negative_alert = self.detect_negative_spike(metrics)
        if negative_alert:
            self.alerts.append(negative_alert)
        
        drop_alert = self.detect_sentiment_drop(df)
        if drop_alert:
            self.alerts.append(drop_alert)
        
        return self.alerts
    
    def print_alerts(self):
        if not self.alerts:
            print("\nStatus: NO CRISIS DETECTED")
            print("All sentiment metrics within normal range")
            return
        
        print(f"\nALERT: {len(self.alerts)} CRISIS INDICATOR(S) DETECTED")
        print("=" * 50)
        
        for i, alert in enumerate(self.alerts, 1):
            print(f"\nAlert {i}:")
            print(f"  Type: {alert['type']}")
            print(f"  Severity: {alert['severity']}")
            print(f"  Message: {alert['message']}")
        
        print("\n" + "=" * 50)
    
    def get_top_negative_keywords(self, df, n=10):
        if 'predicted_sentiment' not in df.columns:
            return None
        
        negative_texts = df[df['predicted_sentiment'] == 'negative']['cleaned_text']
        
        all_words = ' '.join(negative_texts.astype(str)).split()
        
        word_freq = {}
        for word in all_words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {n} Keywords in Negative Tweets:")
        for i, (word, count) in enumerate(sorted_words[:n], 1):
            print(f"  {i}. {word}: {count} mentions")
        
        return sorted_words[:n]


if __name__ == "__main__":
    print("=" * 50)
    print("CRISIS DETECTOR TEST")
    print("=" * 50)
    
    from data_loader import DataLoader
    from text_cleaner import TextCleaner
    from sentiment_analyzer import SentimentAnalyzer
    
    loader = DataLoader()
    cleaner = TextCleaner()
    analyzer = SentimentAnalyzer()
    detector = CrisisDetector()
    
    print("\nLoading and analyzing data...")
    data = loader.load_sentiment140(sample_size=20000)
    
    if data is not None:
        data = loader.prepare_data()
        data = cleaner.clean_dataframe(data)
        data = analyzer.analyze_dataframe(data)
        
        alerts = detector.analyze_for_crisis(data)
        detector.print_alerts()
        detector.get_top_negative_keywords(data, n=10)
        
        print("\n" + "=" * 50)
        print("TEST COMPLETE")
        print("=" * 50)