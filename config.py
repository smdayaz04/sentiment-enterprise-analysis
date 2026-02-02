import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data"
SAVE_PATH = ROOT / "outputs"

os.makedirs(SAVE_PATH, exist_ok=True)

RAW_FILE = DATA_PATH / "sentiment140.csv"
CLEAN_FILE = DATA_PATH / "cleaned_data.csv"

LIMITS = {'pos': 0.05, 'neg': -0.05}
NEG_LIMIT = 40 

BRAND_KEYWORDS = {
    'Airlines': ['flight', 'airline', 'delay', 'cancel', 'airport'],
    'Tech': ['iphone', 'battery', 'update', 'bug', 'crash', 'software'],
    'Finance': ['bank', 'credit', 'loan', 'fee', 'account', 'interest'],
    'Food': ['restaurant', 'food', 'service', 'taste', 'order']
}

SAMPLE_SIZE = None