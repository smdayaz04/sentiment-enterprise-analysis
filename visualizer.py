import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import config

plt.style.use('seaborn-v0_8')

class SentimentVisualizer:
    
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / "figures"
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_sentiment_distribution(self, df, save=True):
        if 'predicted_sentiment' not in df.columns:
            print("No sentiment predictions found")
            return None
        
        sentiment_counts = df['predicted_sentiment'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [config.COLOR_PALETTE.get(s, '#95a5a6') for s in sentiment_counts.index]
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
        
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "sentiment_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig
    
    def plot_sentiment_pie(self, df, save=True):
        if 'predicted_sentiment' not in df.columns:
            print("No sentiment predictions found")
            return None
        
        sentiment_counts = df['predicted_sentiment'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = [config.COLOR_PALETTE.get(s, '#95a5a6') for s in sentiment_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "sentiment_pie.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig
    
    def plot_compound_score_distribution(self, df, save=True):
        if 'vader_compound' not in df.columns:
            print("No compound scores found")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(df['vader_compound'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.axvline(config.SENTIMENT_THRESHOLDS['positive'], color='green', 
                   linestyle='--', linewidth=2, label='Positive Threshold')
        ax.axvline(config.SENTIMENT_THRESHOLDS['negative'], color='red', 
                   linestyle='--', linewidth=2, label='Negative Threshold')
        
        ax.set_xlabel('VADER Compound Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Compound Sentiment Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "compound_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig
    
    def plot_sentiment_over_time(self, df, save=True):
        if 'date' not in df.columns or df['date'].isna().all():
            print("No valid date information found")
            return None
        
        df_time = df.dropna(subset=['date']).copy()
        df_time['date'] = pd.to_datetime(df_time['date'])
        df_time = df_time.sort_values('date')
        
        df_time['date_only'] = df_time['date'].dt.date
        
        daily_sentiment = df_time.groupby(['date_only', 'predicted_sentiment']).size().unstack(fill_value=0)
        daily_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for sentiment in daily_pct.columns:
            color = config.COLOR_PALETTE.get(sentiment, '#95a5a6')
            ax.plot(daily_pct.index, daily_pct[sentiment], 
                   marker='o', label=sentiment.capitalize(), 
                   color=color, linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "sentiment_timeline.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig
    
    def create_wordcloud(self, df, sentiment='negative', save=True):
        if 'predicted_sentiment' not in df.columns or 'cleaned_text' not in df.columns:
            print("Required columns not found")
            return None
        
        sentiment_texts = df[df['predicted_sentiment'] == sentiment]['cleaned_text']
        text = ' '.join(sentiment_texts.astype(str))
        
        if len(text.strip()) == 0:
            print(f"No text available for {sentiment} sentiment")
            return None
        
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='Reds' if sentiment == 'negative' else 'Greens',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Most Common Words in {sentiment.capitalize()} Tweets', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"wordcloud_{sentiment}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close()
        return fig
    
    def create_all_visualizations(self, df):
        print("\nGenerating all visualizations...")
        
        self.plot_sentiment_distribution(df)
        self.plot_sentiment_pie(df)
        self.plot_compound_score_distribution(df)
        self.plot_sentiment_over_time(df)
        self.create_wordcloud(df, sentiment='negative')
        self.create_wordcloud(df, sentiment='positive')
        
        print(f"\nAll visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    print("=" * 50)
    print("VISUALIZER TEST")
    print("=" * 50)
    
    from data_loader import DataLoader
    from text_cleaner import TextCleaner
    from sentiment_analyzer import SentimentAnalyzer
    
    loader = DataLoader()
    cleaner = TextCleaner()
    analyzer = SentimentAnalyzer()
    visualizer = SentimentVisualizer()
    
    print("\nLoading and analyzing data...")
    data = loader.load_sentiment140(sample_size=50000)
    
    if data is not None:
        data = loader.prepare_data()
        data = cleaner.clean_dataframe(data)
        data = analyzer.analyze_dataframe(data)
        
        visualizer.create_all_visualizations(data)
        
        print("\n" + "=" * 50)
        print("TEST COMPLETE")
        print("Check the 'outputs/figures' folder for images")
        print("=" * 50)