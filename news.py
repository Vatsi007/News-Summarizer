import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# ---------------- LOAD ENV ----------------
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



client = OpenAI(api_key=OPENAI_API_KEY,base_url="https://openrouter.ai/api/v1",default_headers={
        "HTTP-Referer": "http://localhost:8000",  # or your Render URL
        "X-Title": "News Summarization Agent"
    })

app = FastAPI(
    title="News Retrieval & Summarization Agent",
    description="Fetches latest news and summarizes it using LLM",
    version="1.0"
)

# ---------------- TOOL 
class SummarizeRequest(BaseModel):
    topic: str
    limit: int | None = 5


def fetch_news(topic: str, limit: int = 5):
    url = "https://newsdata.io/api/1/news"

    params = {
        "apikey": NEWS_API_KEY,
        "q": topic,
        "language": "en",
        "country": "us"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise Exception(
            f"News API failed | status={response.status_code} | response={response.text}"
        )

    data = response.json()
    articles = data.get("results", [])

    return articles[:limit]



# ---------------- AGENT ----------------
def summarize_news(news_text: str, topic: str) -> str:


    prompt = f"""
        You are a professional AI news assistant, similar in tone and clarity to ChatGPT.

        Your job is to analyze news articles and produce responses that are:
        - Clear
        - Well-structured
        - Natural and conversational
        - Easy for a general user to understand

        BEHAVIOR RULES:
        1. Always stay strictly grounded in the provided articles.
        2. Do NOT hallucinate facts or speculate beyond the articles.
        3. If information is uncertain or incomplete, say so clearly.
        4. Do NOT mention sources, APIs, or article metadata.
        5. Do NOT repeat article numbers unless helpful.
        6. Avoid robotic or overly formal language.

        STYLE GUIDELINES:
        - Write like an expert explaining to a curious user.
        - Use smooth transitions between ideas.
        - Prefer short paragraphs over dense blocks of text.
        - If using bullet points, keep them concise and meaningful.
        - If writing paragraphs, keep them flowing and readable.

        STRUCTURE:
        - Start with a brief overview sentence that sets context.
        - Present the most important updates first.
        - Group related points together.
        - End with a concise takeaway if appropriate.

        FORMATTING:
        - Use bullet points only when clarity improves.
        - Otherwise, use natural paragraphs.
        - Do NOT use markdown headings unless clearly helpful.

        FOCUS:
        - Stay tightly focused on the user’s topic.
        - Ignore loosely related or tangential news.
        - Combine similar updates instead of repeating them.

        TONE:
        - Calm
        - Informative
        - Neutral
        - Helpful
        - Confident (but not opinionated)

        Your goal is to make the user feel like:
        “I got a clear, reliable explanation — not just a summary.”

       

        ARTICLES:
        {news_text}

        Please summarize the news for the user.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
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
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/")
def home():
    return {"message": "News Retrieval Agent is running 🚀"}


@app.post("/summarize")
def summarize(request: SummarizeRequest):
    try:
        print("➡️ /summarize called")
        print("Topic:", request.topic)
        print("Limit:", request.limit)
        if not NEWS_API_KEY:
            raise HTTPException(status_code=500, detail="NEWS_API_KEY not configured")

        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        articles = fetch_news(request.topic, request.limit or 5)

        if not articles:
            return {
                "topic": request.topic,
                "summary": "No relevant news articles found."
            }

        combined_text = ""
        for i, article in enumerate(articles, start=1):
            combined_text += f"""
            Article {i}:
            Title: {article.get('title')}
            Description: {article.get('description')}
            """

        summary = summarize_news(combined_text,topic=request.topic)

        return {
            "topic": request.topic,
            "summary": summary
        }

    except Exception as e:
        print("🔥 FATAL ERROR:", repr(e))   # THIS LINE IS CRITICAL
        raise HTTPException(status_code=500, detail=str(e))




