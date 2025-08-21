import streamlit as st
import pandas as pd
import requests
import json
import time
import re
import os
#from dotenv import load_dotenv

# --- Load API Key from .env ---
#load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("API key not found. Please set OPENROUTER_API_KEY in .env file.")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-3.5-turbo"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Utility Functions ---

def analyze_sentiment_or(text):
    """Analyze sentiment using OpenRouter API."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
                You are a sentiment analysis assistant. Your task is to analyze the sentiment of user-provided text.
                For each analysis, you must provide a JSON response with the following keys:
                'sentiment': 'Positive', 'Negative', or 'Neutral'.
                'confidence': a float from 0.0 to 1.0 representing confidence.
                'explanation': a one-sentence explanation.
                'keywords': list of keywords that influenced the sentiment.
                The JSON must be the only output.
                """
            },
            {
                "role": "user",
                "content": f"Analyze the sentiment of the following text:\n\"{text}\"\nReturn only JSON."
            }
        ],
        "response_format": { "type": "json_object" }
    }

    for i in range(5):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                content_string = result['choices'][0]['message']['content']
                parsed_content = json.loads(content_string)
                parsed_content['text'] = text
                parsed_content['confidence'] = parsed_content.get('confidence', 0.95)
                return parsed_content
            else:
                st.error("Invalid response from OpenRouter API.")
                return None
        except requests.exceptions.RequestException as e:
            if hasattr(response, "status_code") and response.status_code == 429:
                st.warning(f"Rate limit exceeded. Retrying in {2 ** i} seconds...")
                time.sleep(2 ** i)
            else:
                st.error(f"API request error: {e}")
                return None
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response from the API: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None
    return None

def extract_keywords_html(text, keywords):
    """Highlights keywords in text using HTML."""
    if not keywords:
        return text
    highlighted_text = text
    for word in sorted(keywords, key=len, reverse=True):
        highlighted_text = re.sub(
            r'\b(' + re.escape(word) + r')\b',
            f'<mark style="background-color: #d1e7dd; padding: 0.2em; border-radius: 0.3em;">\\1</mark>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text

# --- Dashboard Components ---

def show_results(df):
    st.subheader("Analysis Results")
    st.dataframe(df.style.format({'confidence': '{:.2%}'}), use_container_width=True)

    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    sentiment_counts['color'] = sentiment_counts['sentiment'].map({
        'Positive': '#198754',
        'Negative': '#dc3545',
        'Neutral': '#ffc107'
    })
    st.bar_chart(sentiment_counts, x='sentiment', y='count', color='color')

    if not df.empty:
        st.subheader("Detailed Breakdown")
        first_row = df.iloc[0]
        st.markdown(f"**Text:** {first_row['text']}")
        st.markdown(f"**Overall Sentiment:** **{first_row['sentiment']}** with **{first_row['confidence']:.2%}** confidence.")

        st.markdown("---")
        st.markdown("#### Keyword Analysis")
        highlighted_text = extract_keywords_html(first_row['text'], first_row['keywords'])
        st.markdown(f"**Key Drivers:** {highlighted_text}", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Explanation")
        st.markdown(f"**Rationale:** {first_row['explanation']}")

# --- Main App ---

def main():
    st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
    st.title("✨ Sentiment Analysis Dashboard")
    st.markdown("Analyze the emotional tone of text content. Powered by OpenRouter.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Single Text Analysis", "Batch File Upload"])

    with tab1:
        st.subheader("Analyze a single piece of text")
        text_input = st.text_area("Enter your text here:", height=150, placeholder="This product is amazing!")
        if st.button("Analyze Text", key="single_analysis_button"):
            if text_input:
                with st.spinner("Analyzing..."):
                    result = analyze_sentiment_or(text_input)
                if result:
                    df_result = pd.DataFrame([result])
                    show_results(df_result)
            else:
                st.warning("Please enter text.")

    with tab2:
        st.subheader("Analyze multiple texts from a CSV")
        uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])
        if uploaded_file:
            try:
                dataframe = pd.read_csv(uploaded_file)
                st.success(f"File '{uploaded_file.name}' uploaded.")
                if 'text' not in dataframe.columns:
                    st.error("CSV must have a 'text' column.")
                    return
                st.info(f"Found {len(dataframe)} texts.")

                if st.button("Analyze File", key="batch_analysis_button"):
                    analysis_results = []
                    with st.spinner("Processing batch..."):
                        for index, row in dataframe.iterrows():
                            text = str(row['text'])
                            if text.strip():
                                result = analyze_sentiment_or(text)
                                if result:
                                    analysis_results.append(result)
                                else:
                                    st.error("API error. Stopping batch.")
                                    break
                            time.sleep(2)

                    if analysis_results:
                        df_results = pd.DataFrame(analysis_results)
                        show_results(df_results)
                        csv_output = df_results.to_csv(index=False).encode('utf-8')
                        st.markdown("---")
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_output,
                            file_name='sentiment_analysis_results.csv',
                            mime='text/csv'
                        )
            except Exception as e:
                st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
