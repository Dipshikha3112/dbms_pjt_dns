"""Run: python -m unittest test_phase13_planner.py -v"""
import unittest
from phase13_hybrid_query_planner import classify, plan, CostRegistry, CASES

class PlannerTests(unittest.TestCase):
    def test_route_cases(self):
        registry=CostRegistry(path="__phase13_nonexistent_test_costs__.json")
        for question, expected in CASES:
            with self.subTest(question=question):
                self.assertEqual(plan(question, registry)["selected_strategy"], expected)

    def test_no_freeform_patient_disclosure(self):
        from phase13_hybrid_query_planner import sql_execute
        with self.assertRaises(ValueError):
            sql_execute("Show all patient records")

    def test_cost_based_choice(self):
        registry=CostRegistry(path="__phase13_nonexistent_test_costs__.json")
        registry.rows=[]
        for _ in range(3):
            registry.record("sql","sql",100,True)
            registry.record("hybrid","sql",5,True)
        self.assertEqual(plan("How many patients are in the database?",registry)["selected_strategy"],"hybrid")

if __name__=="__main__":
    unittest.main()
