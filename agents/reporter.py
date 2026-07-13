import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def reporter_agent(original_question: str, sub_results: list, causal_result: dict = None):
    summary_input = ""
    for r in sub_results:
        summary_input += f"\nSub-question: {r['question']}\n"
        if r.get("error"):
            summary_input += f"Error: {r['error']}\n"
        else:
            summary_input += f"Columns: {r['columns']}\n"
            summary_input += f"Sample rows: {r['rows'][:8]}\n"

    causal_section = ""
    if causal_result:
        causal_section = f"""
STATISTICAL COMPARISON (region vs region, before/after):
{causal_result['region_summary']}

SIGNIFICANCE TEST (is the target region's drop statistically different from other regions?):
{causal_result['significance_test']}

IMPORTANT: If significant_at_5pct is False, this means the drop is NOT unique to the target region —
it likely reflects a broader, market-wide trend rather than a region-specific issue. Say this explicitly
if that's the case, rather than treating it as a region-specific problem.
"""

    prompt = f"""You are a business analyst writing an executive summary.

Original business question:
"{original_question}"

Here is the data gathered to investigate this question:
{summary_input}
{causal_section}

Write a clear, concise executive report (150-250 words) that:
1. Directly answers the original question based ONLY on the data shown above.
2. If the statistical test shows the drop is NOT region-specific, explicitly say so and reframe the finding.
3. Highlights the most important finding(s).
4. Ends with 1-2 recommended next steps.

Do not invent numbers not present in the data. Be specific and reference actual figures where possible.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()