"""
Test script for FastAPI endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_predict():
    """Test prediction endpoint"""
    print("\n🔮 Testing prediction endpoint...")
    
    # Sample features (adjust based on your model)
    features = {
        "Flow Duration": 120000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        "Flow Bytes/s": 1000.5,
        "Flow Packets/s": 15.2
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"features": features}
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_agent_reason():
    """Test reasoning engine"""
    print("\n🧠 Testing reasoning engine...")
    
    drift_results = {
        "overall_psi": 0.23,
        "n_drifted_features": 5,
        "action": "RETRAIN",
        "feature_psi": {}
    }
    
    context = {
        "time_since_last_retrain": "7 days",
        "retraining_cost": "medium",
        "deployment_risk": "low"
    }
    
    response = requests.post(
        f"{BASE_URL}/agent/reason",
        json={
            "drift_results": drift_results,
            "context": context
        }
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_agent_plan():
    """Test planning agent"""
    print("\n📋 Testing planning agent...")
    
    context = {
        'latest_data_path': 'data/new_batch.csv',
        'base_data_path': 'data/training.csv',
        'holdout_path': 'data/holdout.csv'
    }
    
    response = requests.post(
        f"{BASE_URL}/agent/plan",
        json={
            "goal": "maintain_accuracy_above_0.95",
            "context": context
        }
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_agent_status():
    """Test agent status"""
    print("\n📊 Testing agent status...")
    response = requests.get(f"{BASE_URL}/agent/status")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("🧪 DriftCatcher API Test Suite")
    print("=" * 60)
    
    try:
        test_health()
        test_predict()
        test_agent_reason()
        test_agent_plan()
        test_agent_status()
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running: uv run python api/main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
