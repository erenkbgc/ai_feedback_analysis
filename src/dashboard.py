import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

def run_dashboard(df: pd.DataFrame):
    """
    Runs an interactive Streamlit dashboard for feedback analysis.

    Args:
        df (pd.DataFrame): DataFrame containing processed reviews.
    """
    st.title("AI-Powered Feedback Analysis Dashboard")

    # Sidebar filters
    st.sidebar.header("Filter Options")
    unique_products = df['name'].dropna().unique().tolist()
    product = st.sidebar.selectbox("Select Product", ["All"] + unique_products)
    min_rating, max_rating = int(df['reviews.rating'].min()), int(df['reviews.rating'].max())
    rating = st.sidebar.slider("Rating Range", min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating))
    sentiment = st.sidebar.multiselect("Sentiment", df['sentiment'].unique().tolist(), default=df['sentiment'].unique().tolist())

    # Apply filters
    filtered = df.copy()
    if product != "All":
        filtered = filtered[filtered['name'] == product]
    filtered = filtered[filtered['reviews.rating'].between(rating[0], rating[1])]
    filtered = filtered[filtered['sentiment'].isin(sentiment)]

    st.write(f"### Showing {len(filtered)} Reviews")

    # Sentiment Distribution Pie Chart 
    st.subheader("Sentiment Distribution")
    sent_counts = filtered['sentiment'].value_counts()
    fig_pie = px.pie(
        values=sent_counts.values,
        names=sent_counts.index,
        title="Sentiment Proportion"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Sentiment by Top 10 Products
    st.subheader("Sentiment by Top 10 Products")
    top_products = (
        filtered['name'].value_counts()
        .head(10)
        .index.tolist()
    )
    sentiment_counts = (
        filtered[filtered['name'].isin(top_products)]
        .groupby(['name', 'sentiment'])
        .size()
        .reset_index(name='count')
    )
    fig_bar = px.bar(
        sentiment_counts,
        x='name',
        y='count',
        color='sentiment',
        title='Sentiment Distribution by Top 10 Products'
    )
    fig_bar.update_layout(
        xaxis_tickangle=-35,
        xaxis_title='Product Name',
        yaxis_title='Review Count',
        legend_title='Sentiment',
        height=400,
        margin=dict(l=30, r=30, t=60, b=70)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Keyword word cloud
    st.subheader("Keyword Word Cloud")
    all_keywords = ', '.join(filtered['keywords'].dropna().tolist())
    if all_keywords:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_keywords)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("No keywords found.")

    # Show sample reviews
    st.subheader("Sample Reviews")
    num_show = st.slider("How many reviews to show?", 1, len(filtered), 20)
    st.dataframe(filtered[['reviews.text', 'summary', 'suggestion', 'reviews.rating', 'sentiment', 'keywords', 'name']].head(num_show))

    # Detailed view for one review
    st.subheader("Detailed View")
    if len(filtered) > 0:
        idx = st.number_input("Row number to inspect", min_value=0, max_value=len(filtered)-1, value=0)
        row = filtered.iloc[idx]
        st.markdown(f"**Review:** {row['reviews.text']}")
        st.markdown(f"**Summary:** {row.get('summary', '')}")
        st.markdown(f"**Suggestion:** {row.get('suggestion', '')}")
        st.markdown(f"**Rating:** {row['reviews.rating']} | **Sentiment:** {row['sentiment']}")
        st.markdown(f"**Keywords:** {row['keywords']}")
        st.markdown(f"**Product:** {row['name']}")
    else:
        st.write("No review to display for the selected filters.")
