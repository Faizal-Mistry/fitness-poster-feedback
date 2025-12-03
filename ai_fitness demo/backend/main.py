# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import RepSummary, CoachingResponse
from llm_agent import analyze_rep_with_llm

app = FastAPI(title="AI Fitness Coaching Backend (Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "llm": "dummy"}


@app.post("/analyze_rep", response_model=CoachingResponse)
def analyze_rep(rep: RepSummary):
    rep_dict = rep.dict()
    result = analyze_rep_with_llm(rep_dict)
    return CoachingResponse(**result)
