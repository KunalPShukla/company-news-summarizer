import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from transformers import pipeline
import pandas as pd
import os
import nltk

# Ensure the punkt tokenizer is downloaded
nltk.download('punkt')

# ✅ Simple tokenizer using NLTK
def simple_sent_tokenize(text):
    return nltk.tokenize.sent_tokenize(text)

# ✅ Load translation model
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

# ✅ Fetch news articles
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

# ✅ Summarize content
def summarize_content(content):
    chunks = [content[i:i+3000] for i in range(0, len(content), 3000)]
    summaries = []
    for chunk in chunks:
        summaries.append(chunk[:160])  # Basic summary (first 160 chars)
    return " ".join(summaries)

def generate_summaries(df):
    df["Summary"] = df["Content"].apply(summarize_content)
    return df

# ✅ Sentiment analysis using transformer-based model
def analyze_sentiment(text):
    sentiment_analyzer = pipeline('sentiment-analysis')
    result = sentiment_analyzer(text[:512])[0]
    return result['label']

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(analyze_sentiment)
    return df

# ✅ Combine summaries and translate them to Hindi
def generate_hindi_audio(df):
    # Combine all summaries
    combined_summary_en = " ".join(df["Summary"].tolist())
    
    # Split into ~500 character chunks
    sentences = simple_sent_tokenize(combined_summary_en)
    chunks, current = [], []
    for sentence in sentences:
        if len(" ".join(current + [sentence])) <= 500:
            current.append(sentence)
        else:
            chunks.append(" ".join(current))
            current = [sentence]
    if current:
        chunks.append(" ".join(current))

    # Translate to Hindi using Helsinki-NLP's model
    translated_chunks = [translator(chunk)[0]['translation_text'] for chunk in chunks]
    translated_summary_hi = " ".join(translated_chunks)

    # Generate Hindi TTS
    tts = gTTS(text=translated_summary_hi, lang='hi', slow=False)
    tts.save("summary_hi.mp3")

    return "summary_hi.mp3"

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
        audio_file = generate_hindi_audio(df)

        # Provide Hindi audio output
        if os.path.exists(audio_file):
            st.audio(audio_file, format="audio/mp3")
