import streamlit as st
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import os

# ===================== Helper Functions ===================== #

# Fetch the latest company-related news
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

# ===================== Translation and TTS ===================== #

# Load the translation model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Translate function
def translate_text(text):
    encoded_text = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(encoded_text, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Generate Hindi TTS from translated text
def generate_hindi_audio(text, output_file="summary_hi.mp3"):
    translated_text = translate_text(text)
    tts = gTTS(text=translated_text, lang='hi', slow=False)
    tts.save(output_file)
    return output_file

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
        # Display the dataframe
        st.write("✅ Articles Found!")
        st.dataframe(df[["Title", "Link", "Source", "Date"]])

        # Combine summaries of all articles
        combined_summary = " ".join(df["Content"].tolist())

        # Generate Hindi audio for summaries
        output_file = generate_hindi_audio(combined_summary)

        # Provide Hindi audio output
        if os.path.exists(output_file):
            st.audio(output_file, format="audio/mp3")

