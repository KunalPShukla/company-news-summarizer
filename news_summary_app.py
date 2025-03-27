import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
import os
from transformers import pipeline
import nltk

# ===================== Custom Sentiment Analysis ===================== #

# Download the required NLTK lexicon
nltk.download('vader_lexicon')

# Load pre-trained transformer sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis')

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
        result = sentiment_analyzer(text[:512])[0]
        return result['label']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "neutral"

def generate_sentiments(df):
    df["Sentiment"] = df["Summary"].apply(analyze_sentiment)
    return df

def simple_sent_tokenize(text):
    sentences = text.split(". ")
    return [s.strip() for s in sentences if s]

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
    tts = gTTS(text=hindi_text, lang='hi', slow=False)  # Ensure 'hi' for Hindi language
    tts.save(output_file)

# ===================== Streamlit UI ===================== #

st.title("📈 Company News Sentiment Analyzer (Hindi TTS)")

company = st.text_input("Enter Company Name", value="Tesla")

if st.button("Analyze"):
    st.write(f"🔍 Fetching and analyzing news for **{company}**...")

    # Fetch news
    df = fetch_news(company)
    if df.empty:
        st.error("No articles found.")
    else:
        # Summarize content and analyze sentiment
        df = generate_summaries(df)
        df = generate_sentiments(df)

        st.write("✅ Summarization and Sentiment Analysis Done!")
        st.dataframe(df[["Title", "Summary", "Sentiment"]])

        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        # Generate Hindi audio for summaries
        combined_summary = " ".join(df["Summary"].tolist())
        generate_hindi_audio(combined_summary)

        # Provide Hindi audio output
        if os.path.exists("summary_hi.mp3"):
            st.audio("summary_hi.mp3", format="audio/mp3")
