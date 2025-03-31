import threading
import logging
from flask import Flask, jsonify
import torch 
from transformers import pipeline, set_seed
from flask_cors import CORS
from datetime import datetime
import re
import main_backend  # Import your main backend functionality

from ml.recipe_suggester import suggest_recipes

# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://localhost:5173"]}})

# Set seed for consistent results
set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize model for recipe generation
try:
    generator = pipeline(
        'text-generation',
        model='gpt2-medium',
        device=0 if torch.cuda.is_available() else -1,
        pad_token_id=50256,
        truncation=True
    )
    logger.info(f"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    exit(1)

# Function to start backend
def start_backend():
    try:
        main_backend.run_smart_kitchen()  # Start the backend server
    except Exception as e:
        logger.error(f"Error running backend: {str(e)}")

# Function-based operations for recipe generation
def validate_dish(text, ingredients):
    """Robust dish name validation"""
    if not text or not isinstance(text, str):
        return None

    # Basic cleaning
    text = re.sub(r'[^a-zA-Z ]', '', text).strip()
    words = text.split()

    # Validate length and ingredients
    used_ingredients = [ing for ing in ingredients if ing.lower() in text.lower()]
    return text if (2 <= len(words) <= 8 and len(used_ingredients) >= 2) else None

def generate_fallback_dishes(ingredients):
    """Professional fallback dishes"""
    return [
        f"{ingredients[1]} with Roasted {ingredients[0]} and {ingredients[2]}",
        f"{ingredients[0]} {ingredients[2]} Soup",
        f"Pan-Seared {ingredients[1]} with {ingredients[2]} Glaze"
    ]

@app.route('/api/recommend_dishes', methods=['GET'])
def recommend_dishes():
    start_time = datetime.now()
    try:
        inventory = get_current_inventory()
        ingredients = [item["name"] for item in inventory]

        prompt = f"""Generate exactly 3 professional dish names using ONLY: {', '.join(ingredients)}.
        Format strictly as:
        1. Dish One
        2. Dish Two
        3. Dish Three

        Requirements:
        - Each dish must use at least 2 ingredients
        - No measurements or cooking instructions
        - Professional culinary names only
        - Example output:
        1. Chicken Tomato Roulade
        2. Carrot Ginger Chicken
        3. Roasted Tomato Medley

        Your output:"""

        # Generate with conservative settings
        output = generator(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            do_sample=True
        )[0]['generated_text']

        # Parse and validate
        dishes = []
        for line in output.split('\n'):
            if line.strip() and line[0].isdigit():
                try:
                    dish = validate_dish(line.split('.')[1].strip(), ingredients)
                    if dish:
                        dishes.append(dish)
                except IndexError:
                    continue

        # Ensure exactly 3 valid dishes
        if len(dishes) >= 3:
            final_dishes = dishes[:3]
        else:
            final_dishes = generate_fallback_dishes(ingredients)[:3]

        logger.info(f"Recommendation generated in {(datetime.now() - start_time).total_seconds():.2f}s")
        return jsonify(final_dishes)

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        inventory = get_current_inventory()
        ingredients = [item["name"] for item in inventory]
        return jsonify(generate_fallback_dishes(ingredients))

@app.route('/api/generate_dish', methods=['GET'])
def generate_dish():
    start_time = datetime.now()
    try:
        inventory = get_current_inventory()
        ingredients = [item["name"] for item in inventory]

        prompt = f"""Create ONE professional dish name using ONLY: {', '.join(ingredients)}.
        Requirements:
        - Use at least 2 ingredients
        - Professional restaurant-quality name
        - No measurements or instructions
        - Example: "Chicken Supreme with Tomato Jus"

        Respond ONLY with the dish name:"""

        output = generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50
        )[0]['generated_text']

        # Extract dish name after the prompt
        dish = output.split(prompt)[-1].split('\n')[0].strip()
        validated_dish = validate_dish(dish, ingredients)

        response = {
            "dish": validated_dish or "Chef's Seasonal Special",
            "ingredients": [ing for ing in ingredients if ing.lower() in dish.lower()],
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Dish generated in {(datetime.now() - start_time).total_seconds():.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({
            "dish": "Market Fresh Chicken",
            "ingredients": ["Chicken", "Tomato"],
            "error": "AI generation failed"
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "gpt2-medium",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/suggest_recipes", methods=["GET"])
def suggest_recipes_api():
    recipes = suggest_recipes()
    if recipes is None:
        return jsonify({"message": "No recipes found"}), 404
    return jsonify(recipes), 200

# Main Entry Point to Start Both Backend and Flask Server
if __name__ == '__main__':
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
