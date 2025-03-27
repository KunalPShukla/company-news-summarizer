import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
import re
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ===================== Custom Sentiment Analysis ===================== #

# Extended custom words for sentiment analysis
custom_words = {
    'crushes': 10,
    'beats': 5,
    'misses': -5,
    'trouble': -10,
    'falls': -100,
    'amazing': 20,
    'outstanding': 25,
    'horrible': -20,
    'worst': -25,
    'fantastic': 30,
    'love': 15,
    'hate': -15,
    'good': 10,
    'bad': -10,
    'great': 18,
    'terrible': -18,
    'incredible': 25,
    'excellent': 20,
    'poor': -15,
    'joy': 30,
    'sad': -20,
    'exciting': 15,
    'disappointing': -30,
    'beautiful': 18,
    'depressed': -30,
    'happy': 10,
    'angry': -10,
    'decent': 5,
    'awful': -15,
    'inspiring': 20,
    'miserable': -25,
}

# Instantiate the sentiment intensity analyzer
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
        try:
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
        except Exception:
            pass
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

st.title(" Company News Sentiment Analyzer (Hindi TTS)")

company = st.text_input("Enter Company Name", value="Tesla")

if st.button("Analyze"):
    st.write(f" Fetching and analyzing news for **{company}**...")

    # Fetch news
    df = fetch_news(company)
    if df.empty:
        st.error("No articles found.")
    else:
        # Summarize content and analyze sentiment
        df = generate_summaries(df)
        df = generate_sentiments(df)

        st.write(" Summarization and Sentiment Analysis Done!")
        st.dataframe(df[["Title", "Summary", "Sentiment"]])

        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        # Generate Hindi audio for summaries
        combined_summary = " ".join(df["Summary"].tolist())
        generate_hindi_audio(combined_summary)

        # Provide Hindi audio output
        if os.path.exists("summary_hi.mp3"):
            st.audio("summary_hi.mp3", format="audio/mp3")
