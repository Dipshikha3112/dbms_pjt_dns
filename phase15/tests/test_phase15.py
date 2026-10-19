"""Offline smoke tests; no databases required. Run: python -m unittest discover -s phase15/tests"""
import os, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'phase15'))
os.environ.setdefault('PHASE15_API_KEY','test-key-only')
os.environ.setdefault('PHASE14_API_KEY','test-key-only')
from fastapi.testclient import TestClient
import main

class PortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.client=TestClient(main.app)
    def test_portal_page(self):
        r=self.client.get('/portal');self.assertEqual(r.status_code,200);self.assertIn('Medication Evidence',r.text)
    def test_authentication(self):
        self.assertEqual(self.client.get('/api/v15/activity').status_code,401)
        self.assertEqual(self.client.get('/api/v15/activity',headers={'X-API-Key':'wrong'}).status_code,401)
    def test_activity(self):
        r=self.client.get('/api/v15/activity',headers={'X-API-Key':'test-key-only'});self.assertEqual(r.status_code,200)
    def test_invalid_patient(self):
        r=self.client.get('/api/v15/patient/not-a-uuid',headers={'X-API-Key':'test-key-only'});self.assertEqual(r.status_code,422)
    def test_optimizer(self):
        r=self.client.get('/api/v15/optimizer',headers={'X-API-Key':'test-key-only'});self.assertEqual(r.status_code,200)
        self.assertIn('rows',r.json())
    def test_strategies(self):
        r=self.client.get('/api/v15/strategy-analytics',headers={'X-API-Key':'test-key-only'});self.assertEqual(r.status_code,200)
        self.assertIn('strategies',r.json())

if __name__=='__main__':unittest.main()
