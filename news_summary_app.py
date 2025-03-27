import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
import re
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# ===================== Custom Sentiment Analysis ===================== #

# Updated business-oriented sentiment lexicon
custom_words = {
    'growth': 20,
    'decline': -20,
    'revenue': 15,
    'profit': 25,
    'loss': -30,
    'surge': 30,
    'plunge': -35,
    'merger': 10,
    'acquisition': 15,
    'expansion': 20,
    'restructuring': -15,
    'bankruptcy': -50,
    'funding': 10,
    'investor': 5,
    'stock': 8,
    'IPO': 15,
    'dividend': 12,
    'partnership': 18,
    'collaboration': 20,
    'layoffs': -25,
    'recovery': 25,
    'cutbacks': -20,
    'optimism': 30,
    'pessimism': -30,
    'strategy': 15,
    'performance': 18,
    'outlook': 10,
    'targets': 12,
    'forecast': 15,
    'competition': -10,
    'leadership': 12,
    'productivity': 18,
    'marketshare': 20,
    'shareholder': 10,
    'acquisitions': 15,
    'debt': -20,
    'overperforming': 30,
    'underperforming': -30,
    'mergers': 18,
    'innovation': 25,
    'growth_rate': 20,
    'challenges': -10,
    'recession': -50,
    'demand': 18,
    'supply_chain': 12,
    'restructuring': -15,
    'costs': -10,
    'expenditure': -10,
    'margin': 20,
    'market_position': 15,
    'demand_surge': 25,
    'competitive_edge': 30
}

# Instantiate the sentiment intensity analyzer and update lexicon
vader = SentimentIntensityAnalyzer()
vader.lexicon.update(custom_words)

# ===================== Helper Functions ===================== #

@st.cache_data
def fetch_news(company_name):
    SEARCH_URL = f"https://www.bing.com/news/search?q={company_name}+({'+OR+'.join(['site:indianexpress.com', 'site:businessinsider.com', 'site:financialexpress.com', 'site:firstpost.com'])})&count=100"
    HEADERS = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(SEARCH_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('a', {'class': 'title'})
    data = []

    for i, article in enumerate(articles[:10]):
        title = article.text.strip()
        link = article['href']
        article_page = requests.get(link, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(article_page.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        if content:
            source = soup.find('meta', {'property': 'og:site_name'}) or {}
            date = soup.find('meta', {'property': 'article:published_time'}) or {}
            data.append({
                'Title': title,
                'Link': link,
                'Content': content,
                'Source': source.get('content', 'Unknown'),
                'Date': date.get('content', 'Unknown')
            })
    
    return pd.DataFrame(data)

def summarize_content(content):
    chunks = [content[i:i+3000] for i in range(0, len(content), 3000)]
    summaries = []
    for chunk in chunks:
        summaries.append(chunk[:160])  # For simplicity, using a basic summary (first 160 chars)
    return " ".join(summaries)

def generate_summaries(df):
    df["Summary"] = df["Content"].apply(summarize_content)
    return df

def analyze_sentiment(text):
    try:
        result = vader.polarity_scores(text)
        if result['compound'] >= 0.05:
            return 'positive'
        elif result['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    except:
        return "neutral"

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(analyze_sentiment)
    return df

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

def generate_hindi_audio(text, output_file="summary_hi.mp3"):
    sentences = simple_sent_tokenize(text)
    chunks, current = [], []
    for sentence in sentences:
        if len(" ".join(current + [sentence])) <= 500:
            current.append(sentence)
        else:
            chunks.append(" ".join(current))
            current = [sentence]
    if current:
        chunks.append(" ".join(current))

    # Using gTTS to generate audio in Hindi
    hindi_text = " ".join(chunks)
    tts = gTTS(text=hindi_text, lang='hi', slow=False)
    tts.save(output_file)

# ===================== Streamlit UI ===================== #

st.title("Company News Sentiment Analyzer (Hindi TTS)")

company = st.text_input("Enter Company Name", value="Tesla")

if st.button("Analyze"):
    st.write(f"Fetching and analyzing news for **{company}**...")

    # Fetch news
    df = fetch_news(company)
    if df.empty:
        st.error("No articles found.")
    else:
        # Summarize content and analyze sentiment
        df = generate_summaries(df)
        df = generate_sentiments(df)

        st.write("Summarization and Sentiment Analysis Done!")
        st.dataframe(df[["Title", "Summary", "Sentiment"]])

        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        # Generate Hindi audio for summaries
        combined_summary = " ".join(df["Summary"].tolist())
        generate_hindi_audio(combined_summary)

        # Provide Hindi audio output
        if os.path.exists("summary_hi.mp3"):
            st.audio("summary_hi.mp3", format="audio/mp3")
