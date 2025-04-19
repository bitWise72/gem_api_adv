import requests

response = requests.get(
    '',
    headers={
        'Authorization': 'Bearer YOUR_TOKEN_HERE'
    }
)
print(f"Status code: {response.status_code}")
print(f"Response content: {response.text}")
try:
    print(response.json())
except Exception as e:
    print(f"Error parsing JSON: {e}")

response = requests.post(
    'https://gem-recipe-nopp4bs5f-sayan-gangulys-projects.vercel.app/get_recipe',
    json={
        'user_prompt':'Give me recipe for this choco lava cake',
        'image_url': 'https://res.cloudinary.com/dv28lfhwr/image/upload/v1742749253/cel0fckl0tvypav4q5fy.webp'
    }
)
print(f"\nStatus code: {response.status_code}")
print(f"Response content: {response.text}")
try:
    print(response.json())
except Exception as e:
    print(f"Error parsing JSON: {e}")
