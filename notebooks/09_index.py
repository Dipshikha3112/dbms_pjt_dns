
import argparse
import csv
import getpass
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psycopg


CANDIDATES = [
    ("lab_result", ["patient_id", "result_time DESC"]),
    ("lab_result", ["patient_id", "loinc_code", "result_time DESC"]),
    ("encounter", ["patient_id", "start_time DESC"]),
    ("encounter", ["provider_id", "start_time DESC"]),
    ("diagnosis", ["patient_id", "diagnosis_id"]),
    ("prescription", ["patient_id", "prescription_id"]),
    ("prescription_item", ["prescription_id", "rxcui"]),
    ("patient_procedure", ["patient_id", "procedure_id"]),
    ("claim", ["patient_id", "claim_id"]),
    ("claim", ["payer_id", "patient_id"]),
    ("observation", ["patient_id", "effective_datetime DESC"]),
    ("lab_result", ["loinc_code", "patient_id"]),
]


def safe_query(query):
    query = str(query or "").strip()
    if not query:
        return False

    if not re.match(r"(?is)^SELECT\b", query):
        return False

    forbidden = re.search(
        r"(?is)\b("
        r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|"
        r"GRANT|REVOKE|COPY|CALL|DO|EXECUTE|VACUUM|"
        r"ANALYZE|LOCK|SET|RESET|INTO|FOR\s+UPDATE"
        r")\b",
        query,
    )

    return not forbidden and ";" not in query


def load_workload(path):
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError("Workload CSV has no headers")

        if "query_text" not in reader.fieldnames:
            raise RuntimeError(
                "Expected query_text column. Found: "
                + ", ".join(reader.fieldnames)
            )

        for row in reader:
            query = row.get("query_text", "")

            if safe_query(query):
                rows.append({
                    "query": query.strip(),
                    "family": row.get("query_family", "unknown"),
                    "name": row.get("query_name", "unknown"),
                    "fingerprint": row.get("fingerprint_hash", ""),
                })

    return rows


def explain(cursor, query):
    cursor.execute("EXPLAIN (FORMAT JSON) " + query)
    result = cursor.fetchone()[0]

    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, list):
        result = result[0]

    return result["Plan"]


def plan_cost(plan):
    return float(plan["Total Cost"])


def plan_nodes(plan):
    nodes = []

    def walk(node):
        nodes.append(node)

        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return nodes


def plan_uses_hypothetical(plan):
    for node in plan_nodes(plan):
        index_name = str(node.get("Index Name", ""))

        if re.search(r"<\d+>", index_name):
            return True

    return False


def plan_signature(plan):
    return " -> ".join(
        node.get("Node Type", "?")
        for node in plan_nodes(plan)
    )


def table_relevant(query, table):
    return bool(
        re.search(
            r"(?i)\b(?:FROM|JOIN)\s+"
            r"(?:public\.)?"
            + re.escape(table)
            + r"\b",
            query,
        )
    )


def existing_columns(cursor, table):
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        """,
        (table,),
    )

    return {r[0] for r in cursor.fetchall()}


def index_sql(table, columns):
    parts = []

    for specification in columns:
        tokens = specification.split()
        name = tokens[0]

        direction = (
            " DESC"
            if len(tokens) > 1
            and tokens[1].upper() == "DESC"
            else ""
        )

        parts.append('"' + name + '"' + direction)

    return (
        'CREATE INDEX ON public."'
        + table
        + '" ('
        + ", ".join(parts)
        + ")"
    )


def reset_hypopg(cursor):
    cursor.execute("SELECT hypopg_reset()")


def evaluate_candidate(cursor, table, columns, workload, limit):
    result = {
        "table": table,
        "columns": " | ".join(columns),
        "index_sql": index_sql(table, columns),
        "queries_tested": 0,
        "queries_improved": 0,
        "queries_using_index": 0,
        "total_before_cost": 0.0,
        "total_after_cost": 0.0,
        "weighted_gain_pct": 0.0,
        "max_gain_pct": 0.0,
        "status": "pending",
        "error": "",
    }

    available = existing_columns(cursor, table)
    required = {c.split()[0] for c in columns}

    if not required.issubset(available):
        result["status"] = "missing_columns"
        result["error"] = ",".join(sorted(required - available))
        return result, []

    relevant = [
        row for row in workload
        if table_relevant(row["query"], table)
    ]

    if not relevant:
        result["status"] = "no_relevant_queries"
        return result, []

    groups = defaultdict(list)

    for row in relevant:
        groups[row["family"]].append(row)

    selected = []

    for family, items in sorted(groups.items()):
        selected.extend(items[:limit])

    details = []

    for row in selected:
        query = row["query"]

        try:
            reset_hypopg(cursor)

            before_plan = explain(cursor, query)
            before_cost = plan_cost(before_plan)

            cursor.execute(
                "SELECT * FROM hypopg_create_index(%s)",
                (result["index_sql"],),
            )

            created = cursor.fetchone()

            if not created:
                raise RuntimeError(
                    "HypoPG returned no index"
                )

            after_plan = explain(cursor, query)
            after_cost = plan_cost(after_plan)

            gain = (
                100.0 * (before_cost - after_cost) / before_cost
                if before_cost > 0
                else 0.0
            )

            used = plan_uses_hypothetical(after_plan)

            details.append({
                "table": table,
                "columns": result["columns"],
                "query_family": row["family"],
                "query_name": row["name"],
                "before_cost": round(before_cost, 6),
                "after_cost": round(after_cost, 6),
                "gain_pct": round(gain, 6),
                "hypothetical_index_used": used,
                "before_plan": plan_signature(before_plan),
                "after_plan": plan_signature(after_plan),
                "error": "",
            })

        except Exception as exc:
            details.append({
                "table": table,
                "columns": result["columns"],
                "query_family": row["family"],
                "query_name": row["name"],
                "before_cost": "",
                "after_cost": "",
                "gain_pct": "",
                "hypothetical_index_used": False,
                "before_plan": "",
                "after_plan": "",
                "error": str(exc)[:300],
            })

            # A failed statement may leave the transaction aborted.
            # The caller uses autocommit and each query is isolated.

        finally:
            try:
                reset_hypopg(cursor)
            except Exception:
                pass

    valid = [
        d for d in details
        if d["before_cost"] != ""
    ]

    if not valid:
        result["status"] = "no_valid_explains"
        return result, details

    before = sum(d["before_cost"] for d in valid)
    after = sum(d["after_cost"] for d in valid)

    result.update({
        "queries_tested": len(valid),
        "queries_improved": sum(
            d["gain_pct"] > 0 for d in valid
        ),
        "queries_using_index": sum(
            bool(d["hypothetical_index_used"])
            for d in valid
        ),
        "total_before_cost": round(before, 6),
        "total_after_cost": round(after, 6),
        "weighted_gain_pct": round(
            100 * (before - after) / before, 6
        ) if before > 0 else 0.0,
        "max_gain_pct": max(d["gain_pct"] for d in valid),
        "status": "evaluated",
    })

    return result, details


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="localhost",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5433,
    )

    parser.add_argument(
        "--database",
        default="adaptive_healthcare_phase9_benchmark",
    )

    parser.add_argument(
        "--user",
        default="postgres",
    )

    parser.add_argument(
        "--workload",
        required=True,
    )

    parser.add_argument(
        "--queries-per-family",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default="results/phase9_index_discovery",
    )

    args = parser.parse_args()

    # Hard stop: never target protected PostgreSQL databases.
    if (
        args.port != 5433
        or args.database != "adaptive_healthcare_phase9_benchmark"
        or args.host not in ("localhost", "127.0.0.1")
    ):
        raise RuntimeError(
            "This experiment is restricted to the isolated "
            "Docker benchmark at localhost:5433."
        )

    password = getpass.getpass("Docker PostgreSQL password: ")

    workload = load_workload(args.workload)

    output = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")

    output.mkdir(parents=True, exist_ok=True)

    summary = []
    details = []

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=password,
        autocommit=True,
        options="-c default_transaction_read_only=on "
        "-c statement_timeout=15000",
    ) as conn:

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    current_database(),
                    inet_server_port(),
                    current_setting('port')
                """
            )

            dbname, connected_port, configured_port = cursor.fetchone()

            if dbname != "adaptive_healthcare_phase9_benchmark":
                raise RuntimeError(f"Unsafe database connection: {dbname}")

            if int(configured_port) != 5432:
                raise RuntimeError(
                    f"Unexpected PostgreSQL server port: {configured_port}"
                )

            print("Database:", dbname)
            print("Docker internal PostgreSQL port:", configured_port)
            print("Client connection port:", connected_port)

            cursor.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM pg_extension "
                "WHERE extname='hypopg')"
            )

            if not cursor.fetchone()[0]:
                raise RuntimeError("HypoPG extension is unavailable")

            for number, (table, columns) in enumerate(CANDIDATES, 1):
                print(
                    f"[{number}/{len(CANDIDATES)}] "
                    f"{table}({', '.join(columns)})",
                    flush=True,
                )

                result, records = evaluate_candidate(
                    cursor,
                    table,
                    columns,
                    workload,
                    args.queries_per_family,
                )

                summary.append(result)
                details.extend(records)

                print(
                    "  Status:",
                    result["status"],
                    "| Gain:",
                    result["weighted_gain_pct"],
                    "% | Improved:",
                    result["queries_improved"],
                    "| Index used:",
                    result["queries_using_index"],
                    flush=True,
                )

    summary.sort(
        key=lambda r: r["weighted_gain_pct"],
        reverse=True,
    )

    write_csv(
        output / "candidate_summary.csv",
        summary,
    )

    write_csv(
        output / "query_level_results.csv",
        details,
    )

    positive = [r for r in summary if r["weighted_gain_pct"] > 0]

    report = {
        "database": args.database,
        "port": args.port,
        "workload_queries": len(workload),
        "candidates_tested": len(summary),
        "candidates_with_positive_estimated_gain": len(positive),
        "best_candidate": positive[0] if positive else None,
        "note": (
            "Planner estimates only. No physical indexes created. "
            "Actual latency improvement has not been measured."
        ),
    }

    with open(
        output / "final_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("PHASE 9B — WORKLOAD-AWARE HYPOPG EXPERIMENT")
    print("=" * 72)
    print("Workload queries:", len(workload))
    print("Candidates evaluated:", len(summary))
    print("Candidates with positive estimated gain:", len(positive))

    if positive:
        best = positive[0]

        print(
            "Best candidate:",
            best["table"],
            best["columns"],
        )

        print(
            "Best estimated gain:",
            best["weighted_gain_pct"],
            "%",
        )

    else:
        print(
            "No positive estimated gains found. "
            "Do not force approval."
        )

    print("Results:", output)
    print("=" * 72)


if __name__ == "__main__":
    main()