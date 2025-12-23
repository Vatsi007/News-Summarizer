import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi import HTTPException
from openai import OpenAI

# ---------------- LOAD ENV ----------------
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="News Retrieval & Summarization Agent",
    description="Fetches latest news and summarizes it using LLM",
    version="1.0"
)

# ---------------- TOOL ----------------
def fetch_news(topic: str, limit: int = 5) -> str:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "qInTitle": topic,
        "pageSize": limit,
        "language": "en",
        "sortby":"relevancy",
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = data.get("articles", [])

    if not articles:
        return "No news articles found."

    combined_text = ""
    for i, article in enumerate(articles, start=1):
        combined_text += f"""
Article {i}:
Title: {article.get('title')}
Description: {article.get('description')}
Content: {article.get('content')}
"""

    return combined_text


# ---------------- AGENT ----------------
def summarize_news(news_text: str) -> str:
    prompt = f"""
You are a professional news summarization agent.

Instructions:
- Summarize clearly
- Use bullet points
- Avoid hallucinations
- Keep it concise

News Articles:
{news_text}

Summary:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "News Retrieval & Summarization Agent",
        "version": "1.0"
    }

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "News Retrieval Agent is running 🚀"}


# @app.get("/summarize")
# def summarize(topic: str = Query(default="Ashes cricket 2025", description="News topic to summarize")):
#     news_text = fetch_news(topic)
#     summary = summarize_news(news_text)

#     return {
#         "topic": topic,
#         "summary": summary
#     }



@app.get("/summarize")
def summarize(
    topic: str = Query(
        default="Ashes Cricket ",
        description="News topic to summarize"
    )
):
    if not NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="NEWS_API_KEY not configured")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        news_text = fetch_news(topic)

        if "No news articles found" in news_text:
            return {
                "topic": topic,
                "summary": "No relevant news articles found."
            }

        summary = summarize_news(news_text)

        return {
            "topic": topic,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )
