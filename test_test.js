// Use ESM import syntax
// Save this file with .mjs extension or add "type": "module" to package.json
import fetch from 'node-fetch';

// API base URL
const BASE_URL = 'https://gem-recipe-nopp4bs5f-sayan-gangulys-projects.vercel.app';

// Test the /test endpoint
async function testEndpoint() {
  console.log('Testing /test endpoint:');

  try {
    const response = await fetch(`${BASE_URL}/test`);

    console.log(`Status code: ${response.status}`);
    const json = await response.json();
    console.log(json);
  } catch (error) {
    console.error(`Error making request: ${error.message}`);
  }
}

// Test the /get_recipe endpoint
async function testRecipeEndpoint() {
  console.log('\nTesting /get_recipe endpoint:');

  try {
    const response = await fetch(`${BASE_URL}/get_recipe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_prompt: 'chocolate cake for diabetic friend',
        GEMINI_API_KEY: 'AIzaSyCXrn9vtRUzn0zHueixlfs7qvrVgb5yUwE'  // Include API key from .env
      })
    });

    console.log(`Status code: ${response.status}`);
    // First get the raw text
    const text = await response.text();

    // Try to parse it as JSON
    try {
      const json = JSON.parse(text);
      console.log(JSON.stringify(json, null, 2));
    } catch (e) {
      console.error(`Error parsing JSON: ${e.message}`);
      console.log('Raw response:', text);
    }
  } catch (error) {
    console.error(`Error making request: ${error.message}`);
  }
}

// Run both tests
async function runTests() {
  await testEndpoint();
  await testRecipeEndpoint();
}

runTests(); 