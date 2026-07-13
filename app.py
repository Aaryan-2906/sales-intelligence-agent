import streamlit as st
from graph import run_pipeline
from agents.reporter import reporter_agent
from agents.verifier import verifier_agent
from agents.causal_agent import get_region_comparison

st.set_page_config(page_title="Sales Intelligence Agent", layout="wide")
st.title("Sales Intelligence Agent")
st.caption("Ask a business question. Watch the agents plan, query, and verify before reporting.")

question = st.text_input("Ask a business question:", "Why did sales drop in the South region recently?")

if st.button("Run Analysis"):
    with st.spinner("Planning sub-questions..."):
        results = run_pipeline(question)

    st.subheader("📋 Sub-questions investigated")
    for r in results:
        with st.expander(r["question"]):
            st.code(r["sql"], language="sql")
            if r["error"]:
                st.error(f"Error: {r['error']}")
            else:
                st.write(f"Rows returned: {len(r['rows'])}")
                st.dataframe(r["rows"][:20], use_container_width=True)

    with st.spinner("Running statistical causal comparison..."):
        causal_result = get_region_comparison(
            target_region="PL-South",
            other_regions=["PL-North", "PL-Central"],
            start_date="2024-07-01",
            split_date="2024-10-01",
            end_date="2024-12-31"
        )

    st.subheader("📊 Causal Comparison (Region vs Region)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before/After averages by region:**")
        st.dataframe(causal_result["region_summary"], use_container_width=True)
    with col2:
        st.write("**Statistical significance test:**")
        sig = causal_result["significance_test"]
        if "error" not in sig:
            st.metric("P-value", sig["p_value"])
            if sig["significant_at_5pct"]:
                st.warning("South's drop IS statistically different from other regions — likely region-specific.")
            else:
                st.info("South's drop is NOT statistically different from other regions — likely a market-wide trend.")
        else:
            st.write(sig["error"])

    with st.spinner("Writing report..."):
        report = reporter_agent(question, results, causal_result)

    st.subheader("📝 Executive Report")
    st.write(report)

    st.subheader("✅ Verification")
    for r in results:
        if not r["error"]:
            verdict = verifier_agent(report, r)
            with st.expander(f"Checked against: {r['question']}"):
                st.write(verdict)

    causal_as_data = {
        "columns": ["region", "before_avg", "after_avg", "pct_change"],
        "rows": [(k, v.get("before_avg"), v.get("after_avg"), v.get("pct_change"))
                 for k, v in causal_result["region_summary"].items()],
    }
    causal_verdict = verifier_agent(report, causal_as_data)
    with st.expander("Checked against: Statistical region comparison (causal analysis)"):
        st.write(causal_verdict)