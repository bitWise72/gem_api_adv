import os
import re
import ast
import json
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import logging
import os
import re
import ast
import json
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
from logging import Logger
import requests
from google.generativeai.types import content_types
from google import genai
from google.genai import types

# Simulate Flask test client
# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get API key from environment
gemini_api_key = "AIzaSyCXrn9vtRUzn0zHueixlfs7qvrVgb5yUwE"
DEFAULT_GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not DEFAULT_GEMINI_API_KEY:
    logger.warning("DEFAULT_GEMINI_API_KEY is not set in the environment.")

SYSTEM_PROMPT = """You are a precise and helpful cooking assistant, acting like the voice assistant of Google Gemini, specialized in providing accurate recipe information. Your primary goal is to eliminate vague measurements and ensure baking precision.
The user may request for a recipe for a particular dish or they may provide their own recipe as user request.
If user provides their own recipe, that should be your prime knowledge priority along with online sources to output the below given data

Online recipe platforms often use imprecise measurements like "cups" or "spoons," which can lead to inconsistent baking results. Your role is to provide recipes with ingredient measurements converted to precise grams whenever possible, especially for baking ingredients where accuracy is critical.

When providing recipes:

*   **Measurements in Grams:** Always provide ingredient quantities in grams (g) for solid ingredients and milliliters (ml) for liquids, especially for baking recipes. Avoid vague units like "cups," "tablespoons," and "teaspoons" for ingredients that require precision. If grams are not directly available for certain traditional measurements, clearly state the standardized gram equivalent you are using.
*   **Steps:** Provide clear, step-by-step instructions for preparing the recipe.
*   **Time:** Specify the cooking or preparation time in minutes for each step. If the time is a range, provide both minimum and maximum values (e.g., "8-10 minutes") Provide the time that each step is supposed to take either based on your knowledge or user recipe with higher priority on user recipe.

Structure your response in the following format. Ensure that you strictly adhere to this format so that the response can be easily parsed programmatically:

{
  "step 1": { "procedure": <string>, "measurements": [(ingredient1, measurement1), (ingredient2, measurement2), ...], "time": (min_time, max_time) },
  "step 2": { "procedure": <string>, "measurements": [...], "time": (min_time, max_time) },
  ...
}
"""


def parse_gemini_response(response_text):
    try:

        logger.debug(f"Raw response text to parse: {response_text}")
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
        print(response_text)
        match = re.search(r"\{\s*\"step 1\".*\}", response_text, re.DOTALL)
        if match:
            json_like_str = match.group()
            try:
                # Try using ast.literal_eval first (handles tuples)
                recipe_dict = ast.literal_eval(json_like_str)
            except (SyntaxError, ValueError):
                # Fallback to json.loads if ast fails
                recipe_dict = json.loads(json_like_str.replace("(", "[").replace(")", "]"))
        else:
            print("No JSON object found in response_text.")
            recipe_dict = {}

        if not isinstance(recipe_dict, dict):
            raise ValueError(f"Parsed response is not a dictionary: {type(recipe_dict)}")

        for step, content in recipe_dict.items():
            if not isinstance(content, dict):
                raise ValueError(f"Step {step} content is not a dictionary: {type(content)}")
            required_keys = {"procedure", "measurements", "time"}
            missing_keys = required_keys - set(content.keys())
            if missing_keys:
                raise ValueError(f"Step {step} missing required keys: {missing_keys}")

        return recipe_dict
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}", exc_info=True)
        raise


def get_gemini_response(prompt_text=None, client=None, image_file=None, image_url=None):
    """
    Get a recipe from Gemini API using text and/or image input.

    Parameters:
        prompt_text (str): Text prompt describing the image or recipe idea.
        client: Gemini client instance.
        image_file: File-like object (e.g., from frontend upload).
        image_url (str): URL to the image.

    Returns:
        str: Gemini-generated response or structured JSON error.
    """
    try:
        print(f"Initializing Gemini API...")
        client = genai.Client(api_key=gemini_api_key)

        contents = [SYSTEM_PROMPT, ]

        # Append text if provided
        if prompt_text:
            contents.append(prompt_text)
            print(f"Prompt: {prompt_text[:100]}...")

        # Handle image from URL
        if image_url:
            print(f"Downloading image from URL: {image_url}")
            image_data = requests.get(image_url)
            image_part = types.Part.from_bytes(data=image_data.content, mime_type="image/jpeg")
            contents.append(image_part)

        # Handle image from file-like object
        elif image_file:
            print("Reading uploaded image file...")
            image_bytes = image_file.read()
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            contents.append(image_part)

        if not contents:
            raise ValueError("No prompt_text or image provided to Gemini API.")

        print(f"Sending content to Gemini API: {contents}")
        response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=contents)

        print(f"Raw Gemini response: {response}")
        if not response or not response.text:
            raise ValueError("Empty response from Gemini API")

        recipe_data_dict = parse_gemini_response(response.text)


    except Exception as e:
        print(f"Error in get_gemini_response: {str(e)}")
        return json.dumps({
            "step 1": {
                "procedure": "Error getting recipe from Gemini API. Please try again later.",
                "measurements": [],
                "time": (0, 0)
            }
        })

    print("---------------------------------------------------")
    print(type(recipe_data_dict))
    return recipe_data_dict


get_gemini_response(prompt_text="Give me areceipe of this",
                    image_url="https://res.cloudinary.com/dv28lfhwr/image/upload/v1742749253/cel0fckl0tvypav4q5fy.webp")
