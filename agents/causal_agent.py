import sqlite3
import pandas as pd
from scipy import stats

def get_region_comparison(target_region: str, other_regions: list, start_date: str, split_date: str, end_date: str):
    """
    Compares target_region against other_regions before/after split_date.
    Returns % change for each region, and a t-test on whether target_region's
    drop is statistically different from the others.
    """
    conn = sqlite3.connect("data/sales.db")

    query = """
        SELECT region, date, units_sold
        FROM sales
        WHERE region IN ({})
        AND date >= ? AND date <= ?
    """.format(",".join(["?"] * (len(other_regions) + 1)))

    params = [target_region] + other_regions + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    split = pd.to_datetime(split_date)

    df["period"] = df["date"].apply(lambda d: "before" if d < split else "after")

    daily = df.groupby(["region", "date", "period"])["units_sold"].sum().reset_index()

    summary = {}
    for region in [target_region] + other_regions:
        region_data = daily[daily["region"] == region]
        before = region_data[region_data["period"] == "before"]["units_sold"]
        after = region_data[region_data["period"] == "after"]["units_sold"]

        if len(before) == 0 or len(after) == 0:
            summary[region] = {"error": "insufficient data"}
            continue

        pct_change = ((after.mean() - before.mean()) / before.mean()) * 100

        summary[region] = {
            "before_avg": round(before.mean(), 2),
            "after_avg": round(after.mean(), 2),
            "pct_change": round(pct_change, 2)
        }

    # Statistical test: is target_region's daily units_sold distribution in "after"
    # significantly different from other regions' "after" distribution?
    target_after = daily[(daily["region"] == target_region) & (daily["period"] == "after")]["units_sold"]
    others_after = daily[(daily["region"].isin(other_regions)) & (daily["period"] == "after")]["units_sold"]

    if len(target_after) > 1 and len(others_after) > 1:
        t_stat, p_value = stats.ttest_ind(target_after, others_after, equal_var=False)
        significance = {
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "significant_at_5pct": bool(p_value < 0.05)
        }
    else:
        significance = {"error": "insufficient data for t-test"}

    return {"region_summary": summary, "significance_test": significance}


if __name__ == "__main__":
    result = get_region_comparison(
        target_region="PL-South",
        other_regions=["PL-North", "PL-Central"],
        start_date="2024-07-01",
        split_date="2024-10-01",
        end_date="2024-12-31"
    )
    print(result)