
import getpass
import json
import psycopg

password = getpass.getpass("Docker PostgreSQL password: ")

query = """
SELECT *
FROM public.patient_procedure
WHERE patient_id = (
    SELECT patient_id
    FROM public.patient
    LIMIT 1
)
ORDER BY procedure_id
"""

with psycopg.connect(
    host="localhost",
    port=5433,
    dbname="adaptive_healthcare_phase9_benchmark",
    user="postgres",
    password=password,
    autocommit=True
) as conn:

    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_database(), pg_backend_pid()"
        )
        print("Database/session:", cur.fetchone())

        cur.execute("SELECT hypopg_reset()")

        cur.execute(
            "EXPLAIN (FORMAT JSON) " + query
        )
        baseline = cur.fetchone()[0][0]["Plan"]

        cur.execute(
            """
            SELECT *
            FROM hypopg_create_index(
                'CREATE INDEX ON public.patient_procedure
                 (patient_id, procedure_id)'
            )
            """
        )
        created = cur.fetchone()
        print("Hypothetical index:", created)

        cur.execute(
            "SELECT * FROM hypopg_list_indexes"
        )
        print("Registered indexes:", cur.fetchall())

        cur.execute(
            "EXPLAIN (FORMAT JSON) " + query
        )
        indexed = cur.fetchone()[0][0]["Plan"]

        print("\nBASELINE COST:", baseline["Total Cost"])
        print("INDEXED COST:", indexed["Total Cost"])

        before = float(baseline["Total Cost"])
        after = float(indexed["Total Cost"])

        gain = (
            (before - after) / before * 100
            if before else 0
        )

        print("HYPoPG GAIN:", round(gain, 6), "%")

        print("\nINDEXED PLAN:")
        print(json.dumps(indexed, indent=2))

        cur.execute("SELECT hypopg_reset()")
