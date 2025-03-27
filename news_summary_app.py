import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
import os
import re
from textblob import TextBlob  # Using TextBlob for simple sentiment

# ===================== Helper Functions ===================== #

@st.cache_data
def fetch_news(company_name):
    SEARCH_URL = f"https://www.bing.com/news/search?q={company_name}+({'+OR+'.join(['site:indianexpress.com', 'site:businessinsider.com', 'site:financialexpress.com', 'site:firstpost.com'])})&count=100"
    HEADERS = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(SEARCH_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('a', {'class': 'title'})
    data = []

    for article in articles[:10]:
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
    sentences = content.split('. ')
    return '. '.join(sentences[:3])  # Basic summary: first 3 sentences

def generate_summaries(df):
    df["Summary"] = df["Content"].apply(summarize_content)
    return df

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(analyze_sentiment)
    return df

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

def generate_hindi_audio(text, output_file="summary_hi.mp3"):
    tts = gTTS(text=text, lang='hi', slow=False)
    tts.save(output_file)

# ===================== Streamlit UI ===================== #

st.title("📈 Company News Sentiment Analyzer (Hindi TTS)")

company = st.text_input("Enter Company Name", value="Tesla")

if st.button("Analyze"):
    st.write(f"🔍 Fetching and analyzing news for **{company}**...")

    df = fetch_news_
