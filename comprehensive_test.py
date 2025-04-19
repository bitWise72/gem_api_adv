import requests
import json
import time

BASE_URL = 'https://gem-recipe-nopp4bs5f-sayan-gangulys-projects.vercel.app'

def test_endpoint(name, url, method='GET', data=None):
    print(f"\n===== Testing {name} =====")
    start_time = time.time()
    
    if method == 'GET':
        response = requests.get(url)
    else:
        response = requests.post(url, json=data)
    
    elapsed = time.time() - start_time
    
    print(f"Status code: {response.status_code}")
    print(f"Response time: {elapsed:.2f} seconds")
    
    try:
        json_response = response.json()
        print(f"Response is valid JSON: Yes")
        
        # Pretty print the first part of the response
        formatted_json = json.dumps(json_response, indent=2)
        preview = "\n".join(formatted_json.split("\n")[:20])
        print(f"Response preview:\n{preview}")
        
        if method == 'POST' and 'step 1' in json_response:
            print(f"Recipe format is correct: Yes")
            print(f"Number of steps: {len(json_response)}")
        
        return json_response
    except Exception as e:
        print(f"Response is valid JSON: No - {e}")
        print(f"Raw response: {response.text[:200]}...")
        return None

# Test the basic endpoints
test_endpoint("Home Page", f"{BASE_URL}/")
test_endpoint("Test Endpoint", f"{BASE_URL}/test")

# Test recipe generation with different prompts
recipes = [
    "I need a recipe for chocolate cake",
    "How do I make pasta carbonara?",
    "Give me a recipe for vegan banana bread",
    "I want to make a traditional Indian curry"
]

for i, recipe in enumerate(recipes, 1):
    test_endpoint(
        f"Recipe {i}: {recipe[:30]}...", 
        f"{BASE_URL}/get_recipe", 
        method='POST', 
        data={"user_prompt": recipe}
    )
    # Add a small delay between requests to avoid rate limiting
    time.sleep(2) 