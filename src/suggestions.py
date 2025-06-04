import pandas as pd
from typing import List, Union, Optional

def generate_suggestions_openai(
    texts: List[str],
    api_key: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    batch_size: int = 1,
    prompt_template: Optional[str] = None,
    max_tokens: int = 60
) -> List[str]:
    import openai
    client = openai.OpenAI(api_key=api_key)

    if prompt_template is None:
        prompt_template = (
        "You are an expert customer success analyst. "
        "Read the product review below. "
        "If the review is negative or mentions a problem, suggest one concrete, actionable improvement for the product or customer experience (in 1-2 sentences). "
        "If the review is positive and does NOT mention any problems, suggest how the company can leverage this positive feedback (e.g., thank the customer, use as testimonial, or for marketing) "
        "but do NOT recommend unnecessary changes or random product features. "
        "If the review is neutral, suggest how to increase customer engagement or satisfaction.\n\n"
        "Review: {review}\n\nSuggestion:"
    )

    suggestions = []
    for text in texts:
        prompt = prompt_template.format(review=text)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        suggestion = response.choices[0].message.content.strip()
        suggestions.append(suggestion)
    return suggestions

def add_suggestion_column(
    df: pd.DataFrame,
    text_column: str = 'reviews.text',
    out_column: str = 'suggestion',
    method: str = 'openai',
    openai_api_key: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Adds a suggestion column to the DataFrame.
    """
    df = df.copy()
    if method == 'openai':
        assert openai_api_key is not None, "API key must be provided for OpenAI suggestions."

        # Filter kwargs to only those accepted by generate_suggestions_openai
        allowed = {'model', 'temperature', 'batch_size', 'prompt_template', 'max_tokens'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        df[out_column] = generate_suggestions_openai(
            df[text_column].tolist(),
            api_key=openai_api_key,
            **filtered_kwargs
        )
    else:
        raise ValueError("Currently only OpenAI-based suggestion generation is implemented.")
    return df
