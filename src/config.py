import os

DATA_PATH = "data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
TEXT_COL = "reviews.text"
RATING_COL = "reviews.rating"

SENTIMENT_METHOD = "hf"      # "hf" or "openai"
SENTIMENT_MAXLEN = 512

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SUGGESTION_METHOD = "openai" # "openai"
SUMMARY_METHOD = "openai"    # "openai"
KEYWORDS_METHOD = "keybert"  # "keybert" or "openai"
