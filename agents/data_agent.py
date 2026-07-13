import os
import sqlite3
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SCHEMA = """
Table: sales
Columns:
- date (text, format YYYY-MM-DD) — data ranges from 2022-01-21 to 2024-12-31. Do NOT use DATE('now'); use explicit date literals instead.
- sku (text)
- brand (text)
- segment (text)
- category (text)
- channel (text)
- region (text) — example values: 'PL-Central', 'PL-North', 'PL-South'
- pack_type (text)
- price_unit (float)
- promotion_flag (0 or 1)
- delivery_days (int)
- stock_available (int)
- delivered_qty (int)
- units_sold (int)

IMPORTANT RULES:
- region values are prefixed with "PL-" (e.g. 'PL-South', not 'South'). Always match exact values.
- The dataset's most recent date is 2024-12-31. When a question says "recently" or "last quarter",
  interpret it relative to 2024-12-31, not the current real-world date. Use explicit date literals
  like date >= '2024-10-01' instead of DATE('now', ...).
"""

def get_sql_from_question(question: str) -> str:
    prompt = f"""You are a SQL expert. Given this schema:

{SCHEMA}

Write ONE valid SQLite query to answer this question:
"{question}"

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks.
- Use standard SQLite syntax.
- Table name is 'sales'.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.choices[0].message.content.strip()
    # remove markdown fences if the model adds them anyway
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def run_query(sql: str):
    conn = sqlite3.connect("data/sales.db")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def get_sql_from_question(question: str, previous_error: str = None, previous_sql: str = None) -> str:
    error_context = ""
    if previous_error:
        error_context = f"""
Your previous attempt failed with this SQL:
{previous_sql}

It produced this error:
{previous_error}

Fix the query so it is valid SQLite syntax. Common issue: you cannot use an aggregate
function like SUM() or AVG() directly inside ORDER BY unless it's also in GROUP BY or a subquery alias.
"""

    prompt = f"""You are a SQL expert. Given this schema:

{SCHEMA}

Write ONE valid SQLite query to answer this question:
"{question}"
{error_context}
Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks.
- Use standard SQLite syntax.
- Table name is 'sales'.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def data_agent(question: str, max_retries: int = 2):
    sql = get_sql_from_question(question)
    attempt = 0

    while attempt <= max_retries:
        try:
            columns, rows = run_query(sql)
            return {"sql": sql, "columns": columns, "rows": rows, "error": None, "attempts": attempt + 1}
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                return {"sql": sql, "columns": None, "rows": None, "error": str(e), "attempts": attempt}
            print(f"   Retry {attempt}: fixing SQL error -> {e}")
            sql = get_sql_from_question(question, previous_error=str(e), previous_sql=sql)


if __name__ == "__main__":
    question = "What are the total units sold by region?"
    result = data_agent(question)
    print("\nResult:")
    print(result)