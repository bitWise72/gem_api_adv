from api.get_recipe import app

# This file helps Vercel recognize the main Flask app
# Add error handling
@app.errorhandler(500)
def server_error(e):
    return {"error": "Internal server error. Please check the logs."}, 500