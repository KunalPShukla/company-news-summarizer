import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
import os
import re

# ========== Custom Word-Based Sentiment ========== #
custom_words = {
    # Positive
    'beats': 5, 'soars': 7, 'surges': 6, 'growth': 5, 'record': 6, 'tops': 5,
    'expands': 4, 'profits': 7, 'jumps': 4, 'gains': 5, 'boost': 5, 'crushes': 10,
    'exceeds': 6, 'positive': 4, 'acquisition': 3, 'launch': 3, 'milestone': 4,
    'dominates': 8, 'wins': 6, 'leads': 5, 'rise': 4,

    # Negative
    'misses': -5, 'falls': -8, 'drops': -6, 'decline': -6, 'loss': -7, 'layoffs': -5,
    'downturn': -6, 'negative': -4, 'crisis': -10, 'lawsuit': -6, 'regulatory': -3,
    'trouble': -10, 'plunges': -9, 'bankruptcy': -10, 'fraud': -9, 'investigation': -6,
    'collapse': -10, 'scandal': -8, 'resigns': -4
}

def score_sentiment(text):
    text = text.lower()
    score = 0
    for word, value in custom_words.items():
        if word in text:
            score += value
    if score > 2:
        return "Positive"
    elif score < -2:
        return "Negative"
    else:
        return "Neutral"

# ========== Hindi TTS ========== #
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

    from transformers import pipeline
    translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
    hindi_text = " ".join([translator(chunk)[0]['translation_text'] for chunk in chunks])
    tts = gTTS(text=hindi_text, lang='hi')
    tts.save(output_file)

# ========== News Fetching ========== #
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
    chunks = [content[i:i+3000] for i in range(0, len(content), 3000)]
    summaries = []
    from transformers import pipeline
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
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

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(score_sentiment)
    return df

# ========== Streamlit UI ========== #
st.title("Company News Sentiment Analyzer (Hindi TTS)")

company = st.text_input("Enter Company Name", value="Tesla")

if st.button("Analyze"):
    st.write(f"Fetching and analyzing news for **{company}**...")
    df = fetch_news(company)
    if df.empty:
        st.error("No articles found.")
    else:
        df = generate_summaries(df)
        df = generate_sentiments(df)
        st.write("Summarization and Sentiment Analysis Done!")
        st.dataframe(df[["Title", "Summary", "Sentiment"]])
        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
        combined_summary = " ".join(df["Summary"].tolist())
        generate_hindi_audio(combined_summary)
        if os.path.exists("summary_hi.mp3"):
            st.audio("summary_hi.mp3", format="audio/mp3")
