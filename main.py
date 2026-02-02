from data_loader import DataLoader
from text_cleaner import TextCleaner
from sentiment_analyzer import SentimentAnalyzer
from crisis_detector import CrisisDetector
from visualizer import SentimentVisualizer
import config

def run_pipeline():
    print("🚀 Starting Brand Crisis Detection Pipeline...")
    
    # 1. Load
    loader = DataLoader()
    data = loader.load_sentiment140(sample_size=config.SAMPLE_SIZE)
    df = loader.prepare_data()
    
    # 2. Clean
    cleaner = TextCleaner()
    df = cleaner.clean_dataframe(df)
    
    # 3. Analyze
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataframe(df)
    
    # 4. Detect Crisis
    detector = CrisisDetector()
    detector.analyze_for_crisis(df)
    detector.print_alerts()
    detector.get_top_negative_keywords(df)
    
    # 5. Visualize
    visualizer = SentimentVisualizer()
    visualizer.create_all_visualizations(df)
    
    print("\n✅ Pipeline complete. Results saved in 'outputs/' folder.")
    print("👉 To view the dashboard, run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    run_pipeline()