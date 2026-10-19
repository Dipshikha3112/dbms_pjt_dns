from fastapi.testclient import TestClient
from app import app

client=TestClient(app)

def test_dashboard():
    r=client.get('/')
    assert r.status_code==200
    assert 'Patient workspace' in r.text

def test_docs():
    r=client.get('/openapi.json')
    assert r.status_code==200
    paths=r.json()['paths']
    for path in ['/api/v1/patients/search','/api/v1/patients/{patient_id}/history',
                 '/api/v1/graph/explore','/api/v1/planner/execute','/api/v1/rag/retrieve',
                 '/api/v1/optimizer/recommendations','/api/v1/system/health']:
        assert path in paths

def test_patient_protection():
    assert client.get('/api/v1/patients/search').status_code in (401,503)

def test_invalid_request():
    assert client.post('/api/v1/planner/execute',json={}).status_code in (401,503)
