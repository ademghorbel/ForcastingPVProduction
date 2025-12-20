"""
Test script to see exactly what OpenRouter API returns
"""
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv(override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL")

print(f"Testing OpenRouter API")
print(f"Model: {MODEL}")
print(f"API Key: {API_KEY[:20]}...")
print("=" * 80)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8502",
    "X-Title": "Voltwise"
}

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": "You are an expert energy analyst. Always respond in this exact format:\nRECOMMENDATION: [CHARGE or DISCHARGE or MAINTAIN]\nCONFIDENCE: [High or Medium or Low]\n\nThen provide your detailed analysis."
        },
        {
            "role": "user",
            "content": "Battery is at 30%, clouds at 70%, consumption is 20kWh/day. What should I do?"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}

try:
    print("\nSending request to OpenRouter...")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print("\n" + "=" * 80)
    print("FULL RESPONSE JSON:")
    print("=" * 80)
    result = response.json()
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 80)
    print("EXTRACTED CONTENT:")
    print("=" * 80)
    if 'choices' in result and len(result['choices']) > 0:
        content = result['choices'][0]['message']['content']
        print(content)
        print(f"\nContent length: {len(content)} characters")
        print(f"Content type: {type(content)}")
    else:
        print("ERROR: No choices in response!")
        
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
