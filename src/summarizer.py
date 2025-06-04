import pandas as pd
from typing import List, Union, Optional

def summarize_texts_hf(
    texts: Union[List[str], pd.Series],
    model_name: str = "facebook/bart-large-cnn",
    min_length: int = 10,
    max_length: int = 60,
    batch_size: int = 8,
    device: Optional[str] = None
) -> List[str]:
    """
    Summarizes a list of texts using HuggingFace Transformers pipeline (abstractive).

    Args:
        texts (list-like): List or Series of texts.
        model_name (str): HuggingFace summarization model.
        min_length (int): Minimum length of summary.
        max_length (int): Maximum length of summary.
        batch_size (int): Number of texts to process at once.
        device (str, optional): "cuda", "cpu" or None for auto.

    Returns:
        List[str]: List of summaries.
    """
    from transformers import pipeline
    import torch

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline("summarization", model=model_name, device=device)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            summaries = summarizer(
                list(batch),
                min_length=min_length,
                max_length=max_length,
                truncation=True
            )
            results.extend([x['summary_text'] for x in summaries])
        except Exception:
            results.extend(list(batch))  # Hata olursa orijinal metni bırak
    return results

def summarize_texts_openai(
    texts: List[str],
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.4,
    max_tokens: int = 60,
    prompt_template: Optional[str] = None
) -> List[str]:
    """
    Summarizes a list of texts using OpenAI GPT (new API v1+ compatible).

    Args:
        texts (List[str]): Texts to summarize.
        api_key (str, optional): OpenAI API key. If None, loads from .env.
        model (str): OpenAI model name.
        temperature (float): Model creativity.
        max_tokens (int): Max tokens per response.
        prompt_template (str, optional): Custom prompt template.

    Returns:
        List[str]: List of summaries.
    """
    import os
    from dotenv import load_dotenv

    try:
        import openai
    except ImportError:
        raise ImportError("openai package not found! Run `pip install openai`")

    # Load API key from env if not given
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")

    # Use new-style OpenAI client
    client = openai.OpenAI(api_key=api_key)

    if prompt_template is None:
        prompt_template = (
            "Summarize the following product review in 1-2 sentences:\n\n"
            "{review}\n\nSummary:"
        )

    summaries = []
    for text in texts:
        prompt = prompt_template.format(review=text)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            summary = text  # Hata olursa orijinal metni bırak
        summaries.append(summary)
    return summaries

def add_summary_column(
    df: pd.DataFrame,
    text_column: str = 'reviews.text',
    out_column: str = 'summary',
    method: str = 'hf',
    **kwargs
) -> pd.DataFrame:
    """
    Adds a summary column to a DataFrame using HuggingFace or OpenAI.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column to summarize.
        out_column (str): Output column name.
        method (str): "hf" (HuggingFace) or "openai" (OpenAI).
        kwargs: Additional arguments for summarizer function.

    Returns:
        pd.DataFrame: DataFrame with summary column.
    """
    df = df.copy()
    texts = df[text_column].tolist()
    if method == 'hf':
        df[out_column] = summarize_texts_hf(texts, **kwargs)
    elif method == 'openai':
        df[out_column] = summarize_texts_openai(texts, **kwargs)
    else:
        raise ValueError("Method must be 'hf' or 'openai'.")
    return df
