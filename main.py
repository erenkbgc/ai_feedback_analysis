import streamlit as st
from src.config import (
    DATA_PATH, TEXT_COL, RATING_COL,
    SENTIMENT_METHOD, SENTIMENT_MAXLEN,
    OPENAI_API_KEY, SUGGESTION_METHOD,
    SUMMARY_METHOD, KEYWORDS_METHOD
)
from src.data_loader import load_data
from src.preprocessing import clean_text, clean_ratings
from src.sentiment import add_sentiment_column
from src.keywords import add_keywords_column
from src.summarizer import add_summary_column
from src.suggestions import add_suggestion_column
from src.dashboard import run_dashboard

def main():
    st.set_page_config(page_title="AI Feedback Analysis", layout="wide")
    st.title("AI-Powered Feedback Analysis Dashboard")

    # 1. Load the data (500 samples)
    df = load_data(
        filepath=DATA_PATH,
        sample_size=500
    )
    st.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Clean the rating column and remove NaNs
    df = clean_ratings(df, rating_column=RATING_COL)
    st.success(f"After cleaning ratings: {df.shape[0]} rows left")
    st.write("Unique ratings:", df[RATING_COL].unique())
    st.write("Missing rating count:", df[RATING_COL].isnull().sum())
    st.write("Rating dtype:", df[RATING_COL].dtype)

    df = clean_text(df, text_column=TEXT_COL)

    # 4. Sentiment Analysis
    if SENTIMENT_METHOD == 'openai':
        df = add_sentiment_column(
            df,
            text_column=TEXT_COL,
            method='openai',
            api_key=OPENAI_API_KEY,
            rating_column=RATING_COL,
            hybrid_correction=True
        )
    else:
        df = add_sentiment_column(
            df,
            text_column=TEXT_COL,
            method='hf',
            max_length=SENTIMENT_MAXLEN,
            rating_column=RATING_COL,
            hybrid_correction=True
        )

    # 5. Keyword Extraction
    df = add_keywords_column(
        df,
        text_column=TEXT_COL,
        method=KEYWORDS_METHOD
    )

    # 6. Automatic Summarization (using OpenAI)
    df = add_summary_column(
        df,
        text_column=TEXT_COL,
        method=SUMMARY_METHOD,
        api_key=OPENAI_API_KEY
    )

    # 7. Smart Suggestion Generation (OpenAI only)
    if SUGGESTION_METHOD == 'openai':
        df = add_suggestion_column(
            df,
            text_column=TEXT_COL,
            method='openai',
            openai_api_key=OPENAI_API_KEY,
            rating_column=RATING_COL
        )
    else:
        st.warning("Currently, suggestion generation is only supported with OpenAI!")

    # --- DEBUGGING: Check ratings before dashboard ---
    st.write("DEBUG: Final unique ratings:", df[RATING_COL].unique())
    st.write("DEBUG: Final rating dtype:", df[RATING_COL].dtype)
    st.write("DEBUG: NaN ratings:", df[RATING_COL].isnull().sum())

    # 8. Run the interactive dashboard
    run_dashboard(df)

if __name__ == "__main__":
    main()
