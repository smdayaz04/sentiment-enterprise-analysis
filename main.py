from data_loader import DataLoader
from text_cleaner import TextCleaner
from sentiment_analyzer import SentimentAnalyzer
from crisis_detector import CrisisDetector
from visualizer import SentimentVisualizer
import config

def run_pipeline():
    print("Initializing Enterprise Reputation Intelligence Pipeline...")
    
    loader = DataLoader()
    data = loader.load_sentiment140(sample_size=config.SAMPLE_SIZE)
    df = loader.prepare_data()
    
    cleaner = TextCleaner()
    df = cleaner.clean_dataframe(df)
    
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataframe(df)
    
    detector = CrisisDetector()
    detector.analyze_for_crisis(df)
    detector.print_alerts()
    detector.get_top_negative_keywords(df)
    
    visualizer = SentimentVisualizer()
    visualizer.create_all_visualizations(df)
    
    print("Pipeline execution completed. Artifacts generated in outputs directory.")
    print("Execute streamlit run dashboard/app.py to launch interface.")

if __name__ == "__main__":
    run_pipeline()