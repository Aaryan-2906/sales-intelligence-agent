import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verifier_agent(claim: str, supporting_data: dict):
    """
    claim: a statement the system wants to make (e.g. "sales dropped due to reduced promotions")
    supporting_data: the raw {columns, rows} the claim is supposedly based on
    """
    prompt = f"""You are a strict data verification agent. Your job is to check if a claim
is actually supported by the raw data provided. You must be skeptical and precise.

CLAIM TO VERIFY:
"{claim}"

RAW DATA (columns and rows):
Columns: {supporting_data.get('columns')}
Rows (sample): {supporting_data.get('rows')[:10]}

Answer in this exact format:
VERDICT: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
REASON: [1-2 sentence explanation based only on the data shown]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # quick manual test using fake data
    fake_data = {
        "columns": ["date", "total_units_sold"],
        "rows": [("2024-10-01", 1350), ("2024-10-15", 900), ("2024-10-30", 700)]
    }
    claim = "Units sold in the South region steadily declined through October 2024."
    result = verifier_agent(claim, fake_data)
    print(result)