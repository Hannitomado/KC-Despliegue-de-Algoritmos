from fastapi import APIRouter
from transformers import pipeline

router = APIRouter()

sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

@router.post("/sentiment")
def sentiment(text: str):
    result = sentiment_analyzer(text)
    return {"sentiment": result}

@router.post("/summarize")
def summarize(text: str):
    result = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return {"summary": result[0]["summary_text"]}
