import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
import os
import re
from textblob import TextBlob

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
        sentences = re.split(r'(?<=[.?!])\s+', chunk)
        summary = " ".join(sentences[:3])  # crude summary: first 3 sentences
        summaries.append(summary)
    return " ".join(summaries)

def generate_summaries(df):
    df["Summary"] = df["Content"].apply(summarize_content)
    return df

def analyze_sentiment(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    scores = []
    for chunk in chunks:
        blob = TextBlob(chunk)
        scores.append(blob.sentiment.polarity)

    avg_score = sum(scores) / len(scores) if scores else 0
    if avg_score > 0.1:
        return "positive"
    elif avg_score < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(analyze_sentiment)
    return df

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

@st.cache_resource
def load_translator():
    from transformers import pipeline
    return pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

def generate_hindi_audio(text, output_file="summary_hi.mp3"):
    translator = load_translator()
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

    translated_chunks = [translator(chunk)[0]['translation_text'] for chunk in chunks]
    hindi_text = " ".join(translated_chunks)
    tts = gTTS(text=hindi_text, lang='hi', slow=False)
    tts.save(output_file)

# ===================== Streamlit UI ===================== #

st.title("📈 Company News Sentiment Analyzer (Hindi TTS)")

company = st.text_input("Enter Company Name", value="Tesla")

if st.button("Analyze"):
    st.write(f"🔍 Fetching and analyzing news for **{company}**...")

    df = fetch_news(company)
    if df.empty:
        st.error("No articles found.")
    else:
        df = generate_summaries(df)
        df = generate_sentiments(df)

        st.write("✅ Summarization and Sentiment Analysis Done!")
        st.dataframe(df[["Title", "Summary", "Sentiment"]])

        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        combined_summary = " ".join(df["Summary"].tolist())
        generate_hindi_audio(combined_summary)

        if os.path.exists("summary_hi.mp3"):
            st.audio("summary_hi.mp3", format="audio/mp3")
