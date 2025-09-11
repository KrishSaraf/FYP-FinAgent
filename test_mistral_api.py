#!/usr/bin/env python3
"""
Quick test to verify Mistral API is working properly
"""

import requests
import json

def test_mistral_api():
    """Test Mistral API directly"""
    
    api_key = "5cqXuAMrvlEapMQjZMlJfChoH5npmMs8"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {
                "role": "user",
                "content": "Given that RELIANCE stock has a current price of ₹2,500, 1-day return of +2.5%, RSI of 65, and trading above its 50-day moving average, provide a trading signal in JSON format: {'signal': 'BUY'|'SELL'|'HOLD', 'confidence': 0-100, 'reasoning': 'explanation'}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        print("Testing Mistral API...")
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"Response: {content}")
            
            # Try to parse JSON from response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    print(f"Parsed JSON: {parsed}")
                    return True
                else:
                    print("No JSON found in response")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return False
                
        else:
            print(f"API Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Request error: {e}")
        return False

if __name__ == "__main__":
    success = test_mistral_api()
    print(f"API test {'PASSED' if success else 'FAILED'}")