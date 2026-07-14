import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def reporter_agent(original_question: str, sub_results: list, causal_result: dict = None, forecast_result: dict = None):
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

    forecast_section = ""
    if forecast_result and "error" not in forecast_result:
        forecast_section = f"""
ML FORECAST MODEL RESULTS (XGBoost trained on historical sales, comparing actual vs predicted units_sold):
Number of anomaly days detected (actual deviated significantly from model prediction): {forecast_result['anomaly_count']}
Anomaly dates: {forecast_result['anomaly_dates']}
Mean prediction deviation over this period: {forecast_result['mean_deviation']} (near-zero means the model tracked actual sales closely overall)
Anomaly threshold used: +/- {forecast_result['anomaly_threshold']} units from prediction

IMPORTANT: A small number of anomaly days relative to the total period suggests sales were mostly in line
with normal expected patterns (as predicted by the trained model), with a few specific days deviating.
This distinguishes "the whole period was bad" from "sales were mostly normal, with a few specific outlier days."
"""

    prompt = f"""You are a business analyst writing an executive summary.

Original business question:
"{original_question}"

Here is the data gathered to investigate this question:
{summary_input}
{causal_section}
{forecast_section}

Write a clear, concise executive report (180-280 words) that:
1. Directly answers the original question based ONLY on the data shown above.
2. If the statistical test shows the drop is NOT region-specific, explicitly say so and reframe the finding.
3. Incorporates what the ML forecast model found — specifically whether the drop was a broad sustained pattern or driven by specific anomaly days.
4. Highlights the most important finding(s).
5. Ends with 1-2 recommended next steps.

Do not invent numbers not present in the data. Be specific and reference actual figures where possible.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()