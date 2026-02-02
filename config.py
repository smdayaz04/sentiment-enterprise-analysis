import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data"
SAVE_PATH = ROOT / "outputs"

os.makedirs(SAVE_PATH, exist_ok=True)

# Remove the RAW_FILE line and add this URL instead
DATA_URL = "https://raw.githubusercontent.com/kaz-Anova/Sentiment140/master/training.1600000.processed.noemoticon.csv"

LIMITS = {'pos': 0.05, 'neg': -0.05}
NEG_LIMIT = 40 

BRAND_KEYWORDS = {
    'Airlines': ['flight', 'airline', 'delay', 'cancel', 'airport', 'plane', 'travel'],
    'Tech': ['iphone', 'google', 'apple', 'microsoft', 'phone', 'computer', 'software', 'bug'],
    'Finance': ['bank', 'money', 'cash', 'credit', 'loan', 'pay', 'account', 'rich'],
    'Food': ['food', 'eat', 'dinner', 'restaurant', 'pizza', 'taste', 'burger', 'lunch']
}