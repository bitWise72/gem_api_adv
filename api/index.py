from flask import Flask, request, jsonify
from get_recipe import app as get_recipe_app

app = get_recipe_app

# This file helps Vercel recognize the main Flask app
# Add error handling
@app.errorhandler(500)
def server_error(e):
    return {"error": "Internal server error. Please check the logs."}, 500