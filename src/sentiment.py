import pandas as pd
from typing import List, Optional
import torch

from transformers import pipeline

def hf_sentiment_predict(
    texts,
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    batch_size=32,
    device=None,
    max_length=512
):
    try:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        nlp = pipeline("sentiment-analysis", model=model_name, device=device)
    except NotImplementedError:
        nlp = pipeline("sentiment-analysis", model=model_name, device=-1)

    def truncate(text, n=max_length):
        return text[:n]
    texts = [truncate(str(t)) for t in texts]

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = nlp(list(batch))
        results.extend([x['label'].lower() for x in preds])
    return results

def openai_sentiment_predict(
    texts: List[str],
    api_key: str,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 4,
    temperature: float = 0.0
) -> List[str]:
    """
    Predicts sentiment ("positive", "negative", or "neutral") for a list of texts using OpenAI GPT API.
    """
    import openai
    openai.api_key = api_key
    sentiments = []
    for text in texts:
        prompt = (
            "Classify the sentiment of the following review as strictly 'positive', 'negative', or 'neutral'. "
            "Respond with only one word:\n\n"
            f"Review: {text}\n\nSentiment:"
        )
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = response.choices[0].message.content.strip().lower()
            if "positive" in result:
                sentiments.append("positive")
            elif "negative" in result:
                sentiments.append("negative")
            else:
                sentiments.append("neutral")
        except Exception as e:
            sentiments.append("neutral")
    return sentiments

def add_sentiment_column(
    df: pd.DataFrame,
    text_column: str = 'reviews.text',
    out_column: str = 'sentiment',
    method: str = 'hf',
    api_key: Optional[str] = None,
    max_length: int = 512,
    rating_column: Optional[str] = None,  # Only for hybrid correction
    hybrid_correction: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Adds a sentiment column to a DataFrame using either HuggingFace or OpenAI API.
    Optionally, performs hybrid correction based on rating.
    """
    df = df.copy()
    texts = df[text_column].tolist()

    if method == 'hf':
        hf_args = {}
        for k in ['model_name', 'batch_size', 'device', 'max_length']:
            if k in kwargs:
                hf_args[k] = kwargs[k]
        hf_args['max_length'] = max_length
        df[out_column] = hf_sentiment_predict(texts, **hf_args)
    elif method == 'openai':
        assert api_key is not None, "API key must be provided for OpenAI method."
        df[out_column] = openai_sentiment_predict(texts, api_key=api_key)
    else:
        raise NotImplementedError(f"{method} sentiment method not supported.")

    # HYBRID CORRECTION: Align sentiment with rating if needed
    if hybrid_correction and rating_column and rating_column in df.columns:
        def expected_sentiment(rating):
            if rating >= 4:
                return 'positive'
            elif rating <= 2:
                return 'negative'
            else:
                return 'neutral'
        expected = df[rating_column].apply(expected_sentiment)
        df[out_column] = [
            exp if (exp in ['positive', 'negative'] and sent != exp) else sent
            for sent, exp in zip(df[out_column], expected)
        ]

    return df
