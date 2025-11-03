import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_single_analysis():
    """Test single text analysis"""
    print("\n🔍 Testing Single Text Analysis...")
    
    test_cases = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. I'm very disappointed.",
        "It's okay, nothing special."
    ]
    
    for text in test_cases:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"text": text}
        )
        print(f"\nText: {text}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch_analysis():
    """Test batch text analysis"""
    print("\n🔍 Testing Batch Analysis...")
    
    texts = [
        "Great service and friendly staff!",
        "Worst experience ever. Never again.",
        "Average product, nothing to complain about."
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch-analyze",
        json={"texts": texts}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing Error Handling...")
    
    # Empty text
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"text": ""}
    )
    print(f"\nEmpty text - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Sentiment Analysis API Tests")
    print("=" * 50)
    
    try:
        test_health()
        test_single_analysis()
        test_batch_analysis()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Error: {e}")
