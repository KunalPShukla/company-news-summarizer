
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from gtts import gTTS
import pandas as pd
import torch
import os
import re

# ===================== Helper Functions ===================== #

@st.cache_data
def fetch_news(company_name):
    SEARCH_URL = f"https://www.bing.com/news/search?q={company_name}+({'+OR+'.join(['site:indianexpress.com', 'site:businessinsider.com', 'site:financialexpress.com', 'site:firstpost.com'])})&count=100"
    HEADERS = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(SEARCH_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('a', {'class': 'title'})
    data = []

    for i, article in enumerate(articles[:12]):
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

def clean_text(text):
    text = re.sub(r"[“”‘’\"'`â]", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

def summarize_content(content):
    cleaned = clean_text(content)
    chunks = [cleaned[i:i+3000] for i in range(0, len(cleaned), 3000)]
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=160, min_length=70, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except:
            continue
    return " ".join(summaries)

def generate_summaries(df):
    df["Summary"] = df["Content"].apply(summarize_content)
    return df

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label']
    except:
        return "neutral"

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(analyze_sentiment)
    return df

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

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
