

# app/api_llm.py
from fastapi import APIRouter
from pydantic import BaseModel

from helper_lib.assignment5_llm.llm_base import load_llm, load_tokenizer, generate_answer

router = APIRouter(tags=["LLM"])

tokenizer = load_tokenizer()
llm_model = load_llm(use_finetuned=True)

class TextGenerationRequest(BaseModel):
    question: str
    max_new_tokens: int = 64

@router.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    """
    Assignment 5 LLM endpoint:
    Always returns text in the format:
    'That is a great question. <answer> Let me know if you have any other questions.'
    """
    generated_text = generate_answer(
        llm_model,
        tokenizer,
        question=request.question,
        max_new_tokens=request.max_new_tokens,
    )

    return {"generated_text": generated_text}