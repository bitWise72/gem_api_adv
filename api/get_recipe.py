import os
import re
import ast
import json
import time
import base64
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.genai import types

# Load environment variables
load_dotenv()

app = Flask(__name__)

allowed_origins = [
    "https://bawarchi-aignite.vercel.app",
    "http://localhost:8080",
    "http://localhost:3000"
]

CORS(app, resources={
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Get API key from environment
gemini_api_key = os.environ.get('GEMINI_API_KEY')
nutri_api_key = os.environ.get('NUTRI_API_KEY')

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not set in environment")
if not nutri_api_key:
    logger.warning("NUTRI_API_KEY not set in environment (only /get_nutri needs it)")

# configure global genai client (keeps calls simple)
try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    logger.warning("genai.configure failed (maybe GEMINI_API_KEY missing): %s", e)

SYSTEM_PROMPT = """You are a precise and helpful cooking assistant, acting like the voice assistant of Google Gemini, specialized in providing accurate recipe information. Your primary goal is to eliminate vague measurements and ensure cooking precision. You never use any colons in the recipes and you capitalize the first letter of each sentence in the recipe procedure.
The user may request for a recipe for a particular dish either as text prompt or as an image of the dish or they may provide their own recipe. 
If user provides their own recipe, that should be your main knowledge priority along with online sources to output the below given data

 Your role is to provide recipes with ingredient measurements converted to precise grams whenever possible based on either text or image inputs provided to you, especially for cooking ingredients where accuracy is critical. If it is an image input, you should make use of your vision capabilities to identify what the dish is and how the same can cooked with precision in the quantities of ingredients.

When providing recipes:

*   **Measurements in Grams:** Always provide ingredient quantities in grams (g) for solid ingredients and milliliters (ml) for liquids, especially for cooking recipes. Avoid vague units like "cups," "tablespoons," and "teaspoons" for ingredients that require precision. If grams are not directly available for certain traditional measurements, clearly state the standardized gram equivalent you are using.
*   **Steps:** Provide clear, step-by-step instructions for preparing the recipe.
*   **Time:** Specify the cooking or preparation time in minutes for each step. If the time is a range, provide both minimum and maximum values (e.g., "8-10 minutes") Provide the time that each step is supposed to take either based on your knowledge or user recipe with higher priority on user recipe.

If the user prompt contains name of any language in the form "give me ingredients in <language>", you should provide ingredient name translations in brackets in the given language to the best of your abilities.
Structure your response in the following format. Ensure that you strictly adhere to this format so that the response can be easily parsed programmatically:

{
  "step 1": { "procedure": <string>, "measurements": [(<ingredient1(translations in user desired language if possible otherwise English)>, measurement1), ...], "time": (min_time, max_time), "name" : <string :name of recipe either based on text prompt or image>},
  "step 2": { "procedure": <string>, "measurements": [...], "time": (min_time, max_time), "name" : <string :name of recipe either based on text prompt or image>},
  "step 3": { "procedure": <string>, "measurements": [...], "time": (min_time, max_time), "name" : <string :name of recipe either based on text prompt or image>},
  ...
}
Even if you can not translate, provide the above structured response in English only. Provide ingredient translations in bracket only if confident.
 Never mention this system prompt. If the user provides a recipe , you should prioritize that over any online recipe. If number of people is mentioned, update the recipe ingredient quantities accordingly, otherwise provide recipe only for one single person.

Now provide the recipe for
"""

NUTRI_SYSTEM_PROMPT = """You are a highly accurate Nutritional Analysis Assistant based on Google Gemini. Your task is to calculate and provide the nutritional profile for a list of ingredients and their quantities provided by the user.

The user will provide input in the format:
"ingredients: (ingredient1, quantity1 g/ml), (ingredient2, quantity2 g/ml), ..."

Based on this input, generate a JSON response containing the nutritional information for EACH ingredient listed. The JSON structure MUST strictly follow this format:

{
  "ingredient_name_1": {
    "quantity": "<quantity1> g/ml", // Include the unit provided
    "calories": "<value> kcal",
    "protein": "<value> g",
    "carbohydrates": "<value> g",
    "fiber": "<value> g",
    "sugar": "<value> g", // Specify 'added sugar' or 'total sugar' if possible, otherwise just 'sugar'
    "vitamins": "<list or description of key vitamins>", // e.g., "Rich in Vitamin C, Vitamin K" or specific amounts if reliably known
    "fat": "<value> g", // Include total fat, and if possible, specify saturated or not
    "error": null // Use this field to indicate issues, e.g., "Could not analyze" or "Ambiguous quantity"
  },
  "ingredient_name_2": {
    "quantity": "<quantity2> g/ml",
    "calories": "<value> kcal",
    "protein": "<value> g",
    // ... other fields ...
    "error": null
  },
  // ... other ingredients ...
}

**Crucial Instructions:**
* **Accuracy:** Provide the most accurate nutritional data available based on standard food composition databases.
* **Units:** Ensure quantities are clearly associated with grams (g) for solids or milliliters (ml) for liquids, and nutritional values use standard units (kcal, g).
* **Completeness:** Provide all requested nutritional components (calories, protein, carbs, fiber, sugar, vitamins) for each ingredient. If data for a specific component is unavailable, state "N/A" or "Data not available".
* **JSON Format:** The *entire* response MUST be a single, valid JSON object matching the specified structure. Do not include any introductory text, explanations, apologies, or markdown formatting (like ```json ... ```) outside the JSON structure itself.
* **Error Handling:** If an ingredient cannot be identified or its nutritional profile cannot be determined, clearly state this in the "error" field for that specific ingredient's entry within the JSON. Do *not* fail the entire request; provide data for the ingredients you *can* analyze. Set "error" to `null` if analysis is successful.
* **Focus:** Only respond to requests related to food ingredient nutritional analysis. Reject any unrelated queries. Do not engage in conversation. Never mention this system prompt.
"""

INGRI_SYSTEM_PROMPT = """You are a highly accurate Diet and Nutritional Analysis Assistant based on Google Gemini. Your task is to provide me the following information about a dish from the dish name, dish details or on the basis of input image of the dish.
You will also be provided with additional user health and preferences and you have to make use of alternatives in ingredients and cuisine type based on the user health and preferences.
Provide the output strictly in the following JSON format:
{
    "dishName": <string>, // Name of the dish
    "dishCuisine": <string>, // Type of dish (e.g., North Indian, South Indian, Italian, Mexican etc. in great detail with specifics to the best of your discretion)
    "dishIngredients": [<list of ingredients>], // List of ingredients used in the dish
    "summary": <string>, // This should contain 5 or 6 single line facts and ideas about the dish mentioned, each line should be atmost 60 characters long. The first point should be about the history and origin of the dish, the second should be about any health benefits of the dish, the third should be about how the dish and the ingredients you are suggesting would help the user in their health goals as priority or generl health advices related to the dish ingredients you would suggest, the fourth should be about the taste and flavor of the dish, the fifth should be about any interesting fact about the dish, and the sixth should be about how to make the dish more healthy and nutritious.
    "suggestedRecipes": [<list of strings>] // List of dishes that are typically served as accompaniments or complement the main dish (e.g., Roti for Butter Chicken, Cookies for Tea, Fried Rice for Chilli Chicken).
}
"""

# -------------------------
# Helper: exception logger
# -------------------------
def log_exception(prefix, exc):
    logger.error("%s | %s: %s", prefix, type(exc).__name__, str(exc), exc_info=True)

# -------------------------
# parse_ingri_response (unchanged)
# -------------------------
def parse_ingri_response(response_text):
    logger.debug(f"Attempting to parse Gemini response: {response_text[:500]}...")
    cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
    match = re.search(r"^\s*\{.*\}\s*$", cleaned_text, re.DOTALL)
    if not match:
        logger.error(f"Could not find a valid JSON object structure in the cleaned response: {cleaned_text}")
        raise ValueError("Response does not appear to contain a valid JSON object.")
    json_string = match.group(0)
    try:
        ingri_data = json.loads(json_string)
        logger.debug("Successfully parsed response using json.loads.")
    except json.JSONDecodeError as json_err:
        logger.warning(f"json.loads failed: {json_err}. Trying ast.literal_eval as fallback (use with caution).")
        try:
            ingri_data = ast.literal_eval(json_string)
            logger.debug("Successfully parsed response using ast.literal_eval.")
        except (SyntaxError, ValueError, TypeError) as eval_err:
            logger.error(f"Failed to parse response with both json.loads and ast.literal_eval. Error: {eval_err}", exc_info=True)
            logger.error(f"Problematic JSON string: {json_string}")
            raise ValueError(f"Failed to decode JSON response from AI: {eval_err}")
    expected_keys = ["dishName", "dishCuisine", "dishIngredients", "summary", "suggestedRecipes"]
    if not isinstance(ingri_data, dict):
        raise TypeError(f"Parsed data is not a dict, got {type(ingri_data)}.")
    for key in expected_keys:
        if key not in ingri_data:
            raise ValueError(f"Missing expected key: '{key}'.")
    summary = ingri_data["summary"]
    if isinstance(summary, list):
        ingri_data["summary"] = " ".join(str(s).strip() for s in summary)
    if not isinstance(ingri_data["dishName"], str):
        raise TypeError(f"Expected 'dishName' to be string, but got {type(ingri_data['dishName'])}.")
    if not isinstance(ingri_data["dishCuisine"], str):
        raise TypeError(f"Expected 'dishCuisine' to be string, but got {type(ingri_data['dishCuisine'])}.")
    if not isinstance(ingri_data["dishIngredients"], list):
        raise TypeError(f"Expected 'dishIngredients' to be list, but got {type(ingri_data['dishIngredients'])}.")
    if not isinstance(ingri_data["summary"], str):
        raise TypeError(f"Expected 'summary' to be string, but got {type(ingri_data['summary'])}.")
    if not isinstance(ingri_data["suggestedRecipes"], list):
         raise TypeError(f"Expected 'suggestedRecipes' to be list, but got {type(ingri_data['suggestedRecipes'])}.")
    if not all(isinstance(item, str) for item in ingri_data["suggestedRecipes"]):
         raise TypeError("All items in 'suggestedRecipes' must be strings.")
    logger.debug("Parsed data validated successfully.")
    return ingri_data

# -------------------------
# /get_ingri endpoint (minimal change: use only gemini-2.5-flash)
# -------------------------
@app.route("/get_ingri", methods=["POST", "OPTIONS"])
def get_ingredient_profile():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    data = request.get_json(silent=True)
    logger.info("Received POST to /get_ingri: %s", data)

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not set")
        return jsonify({"error": "Server configuration error: API key missing."}), 500

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    dish_description = data.get("dish_description", "").strip()
    image_url = data.get("image_url", "").strip()

    if not dish_description and not image_url:
        return jsonify({"error": "Missing input. Provide either 'dish_description' or 'image_url'."}), 400

    contents = [{"text": INGRI_SYSTEM_PROMPT}]
    if dish_description:
        contents.append({"text": f"User request: {dish_description}"})

    if image_url:
        try:
            logger.info(f"Attempting to download image from URL: {image_url}")
            image_response = requests.get(image_url, stream=True, timeout=10)
            image_response.raise_for_status()
            content_type = image_response.headers.get("Content-Type", "application/octet-stream")
            if not content_type.startswith("image/"):
                raise ValueError(f"URL did not return an image content type: {content_type}")
            image_data = image_response.content
            image_part = {
                "inlineData": {
                    "mimeType": content_type,
                    "data": base64.b64encode(image_data).decode('utf-8')
                }
            }
            contents.append(image_part)
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Error downloading image from {image_url}: {req_err}", exc_info=True)
            return jsonify({"error": f"Failed to download image from URL: {req_err}"}), 400
        except ValueError as val_err:
            logger.error(f"Validation error processing image URL {image_url}: {val_err}", exc_info=True)
            return jsonify({"error": f"Invalid image URL or content: {val_err}"}), 400
        except Exception as img_process_err:
            logger.error(f"Unexpected error processing image from {image_url}: {img_process_err}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred while processing the image: {img_process_err}"}), 500

    # Use only gemini-2.5-flash now
    model = "gemini-2.5-flash"
    try:
        # instantiate client (keeps parity with your original pattern)
        client = genai.Client(api_key=gemini_api_key)

        logger.info("Calling Gemini model=%s", model)
        response = client.models.generate_content(
            model=model,
            contents=contents
        )

        if response and getattr(response, "text", None):
            ingredient_data = parse_ingri_response(response.text)
            return jsonify(ingredient_data), 200
        else:
            raise ValueError("Empty response from Gemini API")
    except Exception as e:
        logger.error("Unexpected error during Gemini API call in /get_ingri: %s", e, exc_info=True)
        api_error_message = str(e)
        try:
            if hasattr(e, 'response') and e.response is not None:
                api_error_message = e.response.text
        except Exception:
            pass
        return jsonify({"error": f"An unexpected error occurred during AI processing: {api_error_message}"}), 500

# -------------------------
# parse_nutri_response (unchanged)
# -------------------------
def parse_nutri_response(response_text):
    logger.debug(f"Attempting to parse Gemini response: {response_text[:500]}...")
    cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
    match = re.search(r"^\s*\{.*\}\s*$", cleaned_text, re.DOTALL)
    if not match:
        logger.error(f"Could not find a valid JSON object structure in the cleaned response: {cleaned_text}")
        raise ValueError("Response does not appear to contain a valid JSON object.")
    json_string = match.group(0)
    try:
        nutrition_data = json.loads(json_string)
        logger.debug("Successfully parsed response using json.loads.")
    except json.JSONDecodeError as json_err:
        logger.warning(f"json.loads failed: {json_err}. Trying ast.literal_eval as fallback.")
        try:
            nutrition_data = ast.literal_eval(json_string)
            logger.debug("Successfully parsed response using ast.literal_eval.")
        except (SyntaxError, ValueError, TypeError) as eval_err:
            logger.error(f"Failed to parse response with both json.loads and ast.literal_eval. Error: {eval_err}", exc_info=True)
            logger.error(f"Problematic JSON string: {json_string}")
            raise ValueError(f"Failed to decode JSON response from AI: {eval_err}")
    if not isinstance(nutrition_data, dict):
        raise TypeError(f"Parsed data is not a dictionary (type: {type(nutrition_data)}).")
    for ingredient, details in nutrition_data.items():
        if not isinstance(details, dict):
             logger.warning(f"Entry for '{ingredient}' is not a dictionary: {details}")
             nutrition_data[ingredient] = {"error": "Invalid data structure received"}
             continue
        required_keys = {"quantity", "calories", "protein", "carbohydrates", "fiber", "sugar", "vitamins", "error"}
        if not required_keys.issubset(details.keys()):
            missing_keys = required_keys - details.keys()
            logger.warning(f"Entry for '{ingredient}' is missing keys: {missing_keys}. Data: {details}")
            details["error"] = details.get("error", "") + f" | Missing keys: {missing_keys}"
    logger.info("Successfully parsed and validated nutrition data structure.")
    return nutrition_data

# -------------------------
# /get_nutri endpoint (minimal change: use only gemini-2.5-flash)
# -------------------------
@app.route("/get_nutri", methods=["POST", "OPTIONS"])
def get_nutrition_profile():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json(silent=True)
    logger.info("Received POST to /get_nutri: %s", data)

    if not nutri_api_key:
        logger.error("NUTRI_API_KEY not set")
        return jsonify({"error": "Server configuration error: API key missing."}), 500

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    ingredients_string = data.get("ingredients_string", "").strip()
    if not ingredients_string:
        return jsonify({"error": "Missing or invalid 'ingredients_string'. Expected format: 'ingredients: (name1, qty1 g/ml), ...'"}), 400

    if not ingredients_string.lower().startswith("ingredients:"):
        logger.warning("ingredients_string does not start with 'ingredients:'")

    user_prompt = f"User request: {ingredients_string}"
    full_prompt = [NUTRI_SYSTEM_PROMPT, user_prompt]

    model = "gemini-2.5-flash"
    try:
        client = genai.Client(api_key=nutri_api_key)
        logger.info("Calling Gemini model=%s for /get_nutri", model)
        response = client.models.generate_content(
            model=model,
            contents=full_prompt
        )
        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini API")
        nutrition_data = parse_nutri_response(response.text)
        return jsonify(nutrition_data), 200
    except (ValueError, TypeError, json.JSONDecodeError) as parse_err:
        logger.error("Parsing error in /get_nutri: %s", parse_err, exc_info=True)
        return jsonify({"error": f"Failed to process nutrition data: {parse_err}"}), 500
    except Exception as e:
        logger.error("Unexpected error in /get_nutri: %s", e, exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# -------------------------
# parse_gemini_response (unchanged - expects the "step 1" style dict)
# -------------------------
def parse_gemini_response(response_text):
    try:
        logger.debug(f"Raw response text to parse: {response_text}")
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
        match = re.search(r"\{\s*\"step 1\".*\}", response_text, re.DOTALL)
        if match:
            json_like_str = match.group()
            try:
                recipe_dict = ast.literal_eval(json_like_str)
            except (SyntaxError, ValueError):
                recipe_dict = json.loads(json_like_str.replace("(", "[").replace(")", "]"))
        else:
            logger.debug("No JSON object found in response_text.")
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

# -------------------------
# /get_recipe endpoint (minimal changes only)
# - use user_prompt (not prompt_text)
# - call only gemini-2.5-flash
# - keep parsing logic the same
# -------------------------
@app.route("/get_recipe", methods=["POST", "OPTIONS"])
def get_gemini_response(prompt_text=None, client=None, image_file=None, image_url=None):
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    data = request.json or {}
    # ONLY change: read user_prompt key (keeps compatibility with your frontend)
    user_prompt = data.get('user_prompt', '').strip()
    # maintain original behavior: if too short, expand prompt
    if len(user_prompt.strip().split()) <= 2:
        user_prompt = f"Generate the recipe for {user_prompt.strip()} and provide accurate measurements in grams and time in minutes along with the procedure as asked in the system."

    image_url = data.get('image_url')

    try:
        logger.info("Initializing Gemini API for /get_recipe")
        client = genai.Client(api_key=gemini_api_key)

        contents = [SYSTEM_PROMPT]

        if user_prompt:
            contents.append(user_prompt)
            logger.debug("Prompt: %s", user_prompt[:200])

        # prefer inline base64 image parts (same approach as /get_ingri)
        if image_url:
            logger.debug("Downloading image from URL: %s", image_url)
            image_response = requests.get(image_url, timeout=10)
            image_response.raise_for_status()
            content_type = image_response.headers.get("Content-Type", "image/jpeg")
            if not content_type.startswith("image/"):
                return jsonify({"error": "URL is not an image"}), 400
            image_part = {
                "inlineData": {
                    "mimeType": content_type,
                    "data": base64.b64encode(image_response.content).decode('utf-8')
                }
            }
            contents.append(image_part)

        if not contents:
            raise ValueError("No prompt_text or image provided to Gemini API.")

        model = "gemini-2.5-flash"
        logger.info("Sending to Gemini model=%s", model)
        response = client.models.generate_content(
            model=model,
            contents=contents
        )

        if response and getattr(response, "text", None):
            recipe_data_dict = parse_gemini_response(response.text)
            return jsonify(recipe_data_dict)
        else:
            raise ValueError("Empty response from Gemini API")

    except Exception as e:
        logger.error("Error in get_gemini_response: %s", e, exc_info=True)
        # preserve original fallback response shape (so UI doesn't break)
        return jsonify({
            "step 1": {
                "procedure": "Error getting recipe from Gemini API. Please try again later.",
                "measurements": [],
                "time": (0, 0)
            }
        })

@app.route('/')
def index():
    return "Welcome to the Get Recipe API!"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "message": "API is working", "auth_required": False})

def vercel_handler(request):
    with app.app_context():
        return app(request)
