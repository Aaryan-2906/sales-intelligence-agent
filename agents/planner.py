import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def planner_agent(question: str):
    prompt = f"""You are a business analytics planner. A user asked this question:

"{question}"

Break this down into 2-4 specific, data-answerable sub-questions that would help answer it.
Each sub-question should be answerable with a single SQL query against a sales database with columns:
date, sku, brand, segment, category, channel, region, pack_type, price_unit, promotion_flag, delivery_days, stock_available, delivered_qty, units_sold.

Return ONLY a JSON list of strings, no explanation, no markdown. Example format:
["sub-question 1", "sub-question 2", "sub-question 3"]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        sub_questions = json.loads(text)
    except json.JSONDecodeError:
        sub_questions = [text]  # fallback if parsing fails
    return sub_questions


if __name__ == "__main__":
    question = "Why did sales drop in the South region recently?"
    sub_qs = planner_agent(question)
    print("Sub-questions:")
    for i, q in enumerate(sub_qs, 1):
        print(f"{i}. {q}")