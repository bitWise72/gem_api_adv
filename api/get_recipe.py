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

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

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

# ===================== PROMPT =====================
SYSTEM_PROMPT = """
You are a precise cooking assistant.

You MUST return ONLY valid JSON.
Do not include explanations or markdown.

Your output format MUST be exactly:

{
  "recipeName": "<string>",
  "ingredients": [
    {"name": "<string>", "quantity": <number>, "unit": "grams or milliliters"}
  ],
  "procedure": [
    "<Step 1 sentence>",
    "<Step 2 sentence>"
  ],
  "servings": <number>
}

Rules:
- Base the recipe STRICTLY on the user dish name.
- If the user gives "Biryani", return Biryani, never any other dish.
- All quantities must be in grams or milliliters.
- Capitalize the first letter of each sentence in procedure.
- Do not use colons in procedure steps.
- If servings are not given, assume 1.
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
            logger.error("Gemini did NOT return valid JSON. Raw response below:")
            logger.error(text)
            log_exception("JSON parsing failed", e)
            raise ValueError("Model did not return valid JSON")

# ===================== GEMINI CALL =====================
def call_gemini(contents):
    for attempt in range(1, 6):
        try:
            logger.info(f"Calling Gemini | model={MODEL_NAME} | attempt={attempt}")
            logger.debug(f"Prompt contents: {contents}")

            model = genai.GenerativeModel(
                MODEL_NAME,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )

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

# --------------------- RECIPE ---------------------
@app.route("/get_recipe", methods=["POST", "OPTIONS"])
def get_recipe():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True)
    logger.info(f"/get_recipe payload: {data}")

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    prompt_text = data.get("prompt_text", "").strip()
    image_url = data.get("image_url", "").strip()

    if not prompt_text and not image_url:
        return jsonify({"error": "Provide prompt_text or image_url"}), 400

    # ---- STRICTLY BIND MODEL TO USER PROMPT ----
    user_prompt = f"Generate a recipe ONLY for this dish: {prompt_text}"

    contents = [SYSTEM_PROMPT, user_prompt]

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
