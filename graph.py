from agents.planner import planner_agent
from agents.data_agent import data_agent
from agents.reporter import reporter_agent
from agents.verifier import verifier_agent
from agents.causal_agent import get_region_comparison


def run_pipeline(question: str):
    print(f"\n USER QUESTION: {question}\n")

    sub_questions = planner_agent(question)
    print("PLANNER broke this into:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")

    results = []
    for q in sub_questions:
        print(f"\n Running Data Agent for: {q}")
        result = data_agent(q)
        results.append({"question": q, **result})
        if result["error"]:
            print(f"   Error: {result['error']}")
        else:
            print(f"   SQL: {result['sql']}")
            print(f"   Rows returned: {len(result['rows'])}")

    return results


if __name__ == "__main__":
    question = "Why did sales drop in the South region recently?"
    results = run_pipeline(question)

    print("\n\n=== CAUSAL COMPARISON ===\n")
    causal_result = get_region_comparison(
        target_region="PL-South",
        other_regions=["PL-North", "PL-Central"],
        start_date="2024-07-01",
        split_date="2024-10-01",
        end_date="2024-12-31"
    )
    print(causal_result)

    print("\n\n=== FINAL REPORT ===\n")
    report = reporter_agent(question, results, causal_result)
    print(report)

    print("\n\n=== VERIFICATION ===\n")

    # Verify each SQL sub-question's contribution
    for r in results:
        if not r["error"]:
            verdict = verifier_agent(report, r)
            print(f"\nChecked against: {r['question']}")
            print(verdict)

    # Verify the causal/statistical claim separately, against its own data
    causal_as_data = {
        "columns": ["region", "before_avg", "after_avg", "pct_change"],
        "rows": [(k, v.get("before_avg"), v.get("after_avg"), v.get("pct_change"))
                 for k, v in causal_result["region_summary"].items()],
    }
    causal_verdict = verifier_agent(report, causal_as_data)
    print(f"\nChecked against: Statistical region comparison (causal analysis)")
    print(causal_verdict)