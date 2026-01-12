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

# ===================== ENV =====================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NUTRI_API_KEY = os.environ.get("NUTRI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

if not NUTRI_API_KEY:
    raise RuntimeError("NUTRI_API_KEY not set in environment")

# ===================== APP =====================
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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# ===================== PROMPTS =====================
SYSTEM_PROMPT = """You are a precise cooking assistant.
Return recipes strictly as JSON in the requested structure.
Use grams and milliliters only.
Capitalize the first letter of each sentence.
No colons in procedures.
If a recipe is given by user, prioritize it.
If number of people is mentioned, scale quantities.
"""

NUTRI_SYSTEM_PROMPT = """You are a Nutritional Analysis Assistant.
Return strictly valid JSON only.
Format:
{
  "ingredient": {
    "quantity": "100 g",
    "calories": "100 kcal",
    "protein": "2 g",
    "carbohydrates": "20 g",
    "fiber": "3 g",
    "sugar": "5 g",
    "vitamins": "Vitamin C",
    "fat": "1 g",
    "error": null
  }
}
Only analyze food.
"""

INGRI_SYSTEM_PROMPT = """You are a Diet and Food Analysis Assistant.
Return strictly valid JSON only in this format:
{
  "dishName": "",
  "dishCuisine": "",
  "dishIngredients": [],
  "summary": "",
  "suggestedRecipes": []
}
Use user health preferences if given.
"""

# ===================== LOGGING =====================
def log_exception(prefix, e):
    logger.error(
        f"{prefix} | Type={type(e).__name__} | Message={str(e)}",
        exc_info=True
    )

# ===================== JSON HELPERS =====================
def clean_json(text):
    text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return match.group(0)

def parse_json_response(text):
    try:
        logger.debug(f"Attempting json.loads on: {text[:500]}")
        return json.loads(clean_json(text))
    except json.JSONDecodeError:
        logger.warning("json.loads failed, trying ast.literal_eval")
        try:
            return ast.literal_eval(clean_json(text))
        except Exception as e:
            log_exception("JSON parsing failed", e)
            raise

# ===================== GEMINI CALL =====================
def call_gemini(contents):
    for attempt in range(1, 6):
        try:
            logger.info(f"Calling Gemini | model={MODEL_NAME} | attempt={attempt}")
            logger.debug(f"Prompt contents: {contents}")

            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(contents)

            if not response or not getattr(response, "text", None):
                raise ValueError("Empty response from Gemini")

            logger.debug(f"Raw Gemini response: {response.text}")
            return response.text

        except Exception as e:
            log_exception(f"Gemini failure | attempt={attempt}", e)

            msg = str(e).lower()
            if "503" in msg or "overloaded" in msg or "quota" in msg or "rate" in msg:
                wait = 2 ** attempt
                logger.warning(f"Retrying Gemini in {wait}s")
                time.sleep(wait)
                continue

            raise

    raise RuntimeError("Gemini failed after maximum retries")

# ===================== ROUTES =====================
@app.route("/", methods=["GET"])
def index():
    return "Welcome to the Get Recipe API!"

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "API is working", "auth_required": False})

# --------------------- INGREDIENT PROFILE ---------------------
@app.route("/get_ingri", methods=["POST", "OPTIONS"])
def get_ingri():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True)
    logger.info(f"/get_ingri payload: {data}")

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    dish_description = data.get("dish_description", "").strip()
    image_url = data.get("image_url", "").strip()

    if not dish_description and not image_url:
        return jsonify({"error": "Provide dish_description or image_url"}), 400

    contents = [{"text": INGRI_SYSTEM_PROMPT}]

    if dish_description:
        contents.append({"text": dish_description})

    if image_url:
        try:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()

            if not r.headers.get("Content-Type", "").startswith("image/"):
                return jsonify({"error": "URL is not an image"}), 400

            image_part = {
                "inlineData": {
                    "mimeType": r.headers["Content-Type"],
                    "data": base64.b64encode(r.content).decode()
                }
            }
            contents.append(image_part)
        except Exception as e:
            log_exception("Image download failed", e)
            return jsonify({"error": str(e)}), 400

    try:
        raw_text = call_gemini(contents)
        data = parse_json_response(raw_text)
        return jsonify(data), 200
    except Exception as e:
        log_exception("/get_ingri failed", e)
        return jsonify({"error": str(e)}), 500

# --------------------- NUTRITION PROFILE ---------------------
@app.route("/get_nutri", methods=["POST", "OPTIONS"])
def get_nutri():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True)
    logger.info(f"/get_nutri payload: {data}")

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    ingredients_string = data.get("ingredients_string", "").strip()
    if not ingredients_string:
        return jsonify({"error": "ingredients_string missing"}), 400

    contents = [NUTRI_SYSTEM_PROMPT, ingredients_string]

    try:
        raw_text = call_gemini(contents)
        nutrition_data = parse_json_response(raw_text)
        return jsonify(nutrition_data), 200
    except Exception as e:
        log_exception("/get_nutri failed", e)
        return jsonify({"error": str(e)}), 500

# --------------------- RECIPE ---------------------
@app.route("/get_recipe", methods=["POST", "OPTIONS"])
def get_recipe():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True)
    logger.info(f"/get_recipe payload: {data}")

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    user_prompt = data.get("user_prompt", "").strip()
    image_url = data.get("image_url", "").strip()

    contents = [SYSTEM_PROMPT]

    if user_prompt:
        contents.append(user_prompt)

    if image_url:
        try:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()

            if not r.headers.get("Content-Type", "").startswith("image/"):
                return jsonify({"error": "URL is not an image"}), 400

            image_part = {
                "inlineData": {
                    "mimeType": r.headers["Content-Type"],
                    "data": base64.b64encode(r.content).decode()
                }
            }
            contents.append(image_part)
        except Exception as e:
            log_exception("Image processing failed", e)
            return jsonify({"error": str(e)}), 400

    try:
        raw_text = call_gemini(contents)
        recipe_data = parse_json_response(raw_text)
        return jsonify(recipe_data), 200
    except Exception as e:
        log_exception("/get_recipe failed", e)
        return jsonify({"error": str(e)}), 500

# ===================== VERCEL HANDLER =====================
def vercel_handler(request):
    with app.app_context():
        return app(request)
