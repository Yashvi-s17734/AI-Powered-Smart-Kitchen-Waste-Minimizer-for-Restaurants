from flask import Flask, jsonify
from transformers import pipeline, set_seed
import re
from flask import request
from ml.stock_predictor import predict_optimal_stock
import pandas as pd
from ml.usage_tracker import predict_usage_rate,suggest_reorder
import os
from werkzeug.utils import secure_filename
from ml.spoilage_detector import detect_spoilage
from flask_cors import CORS
import torch
import logging
from ml.recipe_suggester import suggest_recipes_1
app = Flask(__name__)


CORS(app, resources={r"/api/*": {
    "origins": ["http://localhost:3000", "http://localhost:5000", "http://localhost:5173", "http://localhost:5001"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})
from datetime import datetime
set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'  # Adjust this path as needed
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize model with production-ready settings
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

def get_current_inventory():
    """Simulate inventory fetch - replace with your actual DB call"""
    return [
        {"name": "Tomato", "quantity": 10, "expires": "2023-12-31"},
        {"name": "Chicken", "quantity": 5, "expires": "2023-12-25"}, 
        {"name": "Carrot", "quantity": 6, "expires": "2024-01-05"}
    ]

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
def suggest_recipes():
    try:
        recipe = suggest_recipes_1()
        if not recipe:
            return jsonify({"message": "No recipes found"}), 404

        # Format recipes in the expected response format
        formatted_recipes = [
            {
                "name": recipe["name"], 
                "score": recipe["match_score"], 
                "missing": recipe["missing_ingredients"]
            }
            
        ]
        # for recipe in recipe:
        return jsonify(formatted_recipes), 200
    except Exception as e:
        return jsonify({"message": f"Error fetching recipes: {str(e)}"}), 500
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/api/detect_spoilage', methods=['POST'])
def detect_spoilage_route():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the directory exists before saving the file
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file.save(file_path)

        item_name = request.form.get('item_name', 'Unknown')

        # Call your spoilage detection function
        is_spoiled, spoilage_percentage = detect_spoilage(file_path, item_name)

        return jsonify({
            "is_spoiled": is_spoiled,
            "spoilage_percentage": spoilage_percentage,
            "item_name": item_name
        }), 200

    return jsonify({"message": "File type not allowed"}), 400




@app.route('/api/predict_stock', methods=['GET'])
def predict_stock():
    item_name = request.args.get('item_name')
    print(item_name)
    if not item_name:
        return jsonify({"message": "Missing item_name parameter"}), 400

    optimal_stock = predict_optimal_stock(item_name)
    if optimal_stock is None:
        return jsonify({"message": f"Could not predict stock for {item_name}"}), 404

    return jsonify({
        "item_name": item_name,
        "optimal_stock": optimal_stock
    }), 200







# API endpoint to track usage
@app.route('/api/track_usage', methods=['POST'])
def track_usage():
    try:
        data = request.get_json()
        item_name = data.get('item_name')
        quantity_used = data.get('quantity_used')

        if not item_name or not quantity_used:
            return jsonify({"error": "Item name and quantity used are required"}), 400

        quantity_used = float(quantity_used)  # Ensure float

        file_path = 'ml/data/usage_data.csv'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            columns = ["item_name", "usage_rate", "quantity_used", "date"]
            df = pd.DataFrame(columns=columns)
            df.to_csv(file_path, index=False)

        current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        new_entry = {
            "item_name": item_name,
            "usage_rate": 0,
            "quantity_used": quantity_used,
            "date": current_date
        }

        data = pd.read_csv(file_path)
        data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
        data.to_csv(file_path, index=False)

        logger.info(f"Usage data for {item_name} added successfully")
        return jsonify({"message": "Usage data added successfully!"}), 200
    except ValueError as e:
        logger.error(f"Invalid quantity_used value: {str(e)}")
        return jsonify({"error": "Quantity used must be a number"}), 400
    except Exception as e:
        logger.error(f"Error in track_usage: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# API endpoint to get usage rate
@app.route('/api/get_usage_rate', methods=['GET'])
def get_usage_rate():
    try:
        item_name = request.args.get('item_name')
        if not item_name:
            return jsonify({"error": "Item name is required"}), 400

        logger.info(f"Fetching usage rate for {item_name}")
        usage_rate, status = predict_usage_rate(item_name)
        logger.info(f"predict_usage_rate returned: rate={usage_rate}, status={status}")

        if status != "Success":
            return jsonify({"error": f"Could not find usage data for {item_name}"}), 404

        if not isinstance(usage_rate, (int, float)):
            logger.error(f"Usage rate is not numeric: {usage_rate} (type: {type(usage_rate)})")
            return jsonify({"error": "Internal error: usage rate is invalid"}), 500

        return jsonify({"item_name": item_name, "usage_rate": round(usage_rate, 2)}), 200
    except Exception as e:
        logger.error(f"Error in get_usage_rate: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# API endpoint to suggest reorder
@app.route('/api/suggest_reorder', methods=['POST'])
def suggest_reorder_endpoint():
    try:
        data = request.get_json()
        item_name = data.get('item_name')
        quantity = data.get('quantity')

        if not item_name or not quantity:
            return jsonify({"error": "Item name and quantity are required"}), 400

        message, status = suggest_reorder(item_name, quantity)
        if status != "Success":
            return jsonify({"error": message}), 400

        return jsonify({"message": message}), 200
    except Exception as e:
        logger.error(f"Error in suggest_reorder_endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
    