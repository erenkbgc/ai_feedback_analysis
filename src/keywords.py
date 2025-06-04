
import pandas as pd
from typing import List, Union, Optional

def extract_keywords_keybert(
    texts: Union[List[str], pd.Series],
    model_name: str = "all-MiniLM-L6-v2",
    top_n: int = 5
) -> List[str]:
    """
    Extracts top N keywords per text using KeyBERT.

    Args:
        texts (list-like): List or Series of texts.
        model_name (str): Sentence-transformer model name.
        top_n (int): Number of keywords per text.

    Returns:
        List[str]: List of comma-separated keyword strings.
    """
    from keybert import KeyBERT
    kw_model = KeyBERT(model_name)
    keywords_all = []
    for text in texts:
        keywords = kw_model.extract_keywords(text, top_n=top_n)
        keywords_flat = [kw for kw, _ in keywords]
        keywords_all.append(", ".join(keywords_flat))
    return keywords_all

def add_keywords_column(
    df: pd.DataFrame,
    text_column: str = 'reviews.text',
    out_column: str = 'keywords',
    method: str = 'keybert',
    **kwargs
) -> pd.DataFrame:
    """
    Adds a keywords column to DataFrame using specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column to analyze.
        out_column (str): Output column name.
        method (str): 'keybert' or 'bertopic'.
        kwargs: Additional args for extractor function.

    Returns:
        pd.DataFrame: DataFrame with keywords.
    """
    df = df.copy()
    texts = df[text_column].tolist()
    if method == 'keybert':
        df[out_column] = extract_keywords_keybert(texts, **kwargs)
    elif method == 'bertopic':
        df[out_column] = extract_topics_bertopic(texts, **kwargs)
    else:
        raise ValueError("Method must be 'keybert' or 'bertopic'")
    return df

def extract_topics_bertopic(
    texts: Union[List[str], pd.Series],
    top_n_words: int = 3
) -> List[str]:
    """
    Uses BERTopic to extract dominant topic words for each text.

    Args:
        texts (list-like): List or Series of texts.
        top_n_words (int): Number of words per topic.

    Returns:
        List[str]: Topic words string per review.
    """
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    topic_words_map = {
        row['Topic']: topic_model.get_topic(row['Topic'])[:top_n_words]
        for _, row in topic_info.iterrows()
        if row['Topic'] != -1
    }
    results = []
    for t in topics:
        if t in topic_words_map:
            results.append(", ".join([w for w, _ in topic_words_map[t]]))
        else:
            results.append("")
    return results
