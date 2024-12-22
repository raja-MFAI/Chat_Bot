import os
from pymongo import MongoClient
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from bson.objectid import ObjectId
import torch
from flask_cors import CORS

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check environment variables
if not MONGO_URI or not HUGGINGFACE_API_KEY:
    raise EnvironmentError("MONGO_URI or HUGGINGFACE_API_KEY not set in environment.")

# Load the DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["test"]
products_collection = db["products"]

# Enable CORS for the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store limited chat history for context
chat_history_ids = None
last_product_data = None

# Route to interact with the chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    global chat_history_ids, last_product_data

    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Identify attribute from the query
    attribute = next((keyword for keyword in ["price", "material", "description", "category", "condition"] if keyword in user_query.lower()), None)

    # Extract potential product name from the query
    words = user_query.lower().split()
    product_name = None
    for word in words:
        product_data = products_collection.find_one({"name": {"$regex": word, "$options": "i"}})
        if product_data:
            product_name = product_data.get("name")
            last_product_data = product_data
            break

    if not last_product_data:
        return jsonify({"response": "Sorry, I couldn't find a matching product for your query."}), 200

    if attribute:
        response = {
            "price": f"The price of {last_product_data['name']} is {last_product_data.get('price', 'unknown')}.",
            "material": f"Material information is not available for {last_product_data['name']} in our database.",
            "description": f"Description of {last_product_data['name']}: {last_product_data.get('description', 'No description available')}",
            "category": f"The category of {last_product_data['name']} is {last_product_data.get('category', 'unknown')}",
            "condition": f"The condition of {last_product_data['name']} is {last_product_data.get('condition', 'unknown')}"
        }.get(attribute, "Sorry, I couldn't retrieve that information.")
        return jsonify({"response": response}), 200

    # Generate a fallback response using the chatbot model
    new_user_input_ids = tokenizer.encode(user_query + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    chat_history_ids = chat_history_ids[:, -1000:]

    return jsonify({"response": response}), 200

# Route to create a new product
@app.route("/products", methods=["POST"])
def create_product():
    product_data = request.json
    if not product_data or "name" not in product_data:
        return jsonify({"error": "Product data is incomplete"}), 400

    result = products_collection.insert_one(product_data)
    return jsonify({"message": "Product created successfully", "product_id": str(result.inserted_id)}), 201

# Route to retrieve all products
@app.route("/products", methods=["GET"])
def get_products():
    products = list(products_collection.find())
    response = [
        {
            "product_id": str(product["_id"]),
            "name": product["name"],
            "price": product.get("price"),
            "category": product.get("category"),
            "condition": product.get("condition"),
            "images": product.get("images", []),
            "description": product.get("description"),
            "user": str(product.get("user")) if product.get("user") else None,
        }
        for product in products
    ]
    return jsonify({"products": response}), 200

# Route to update a product
@app.route("/products/<product_id>", methods=["PUT"])
def update_product(product_id):
    update_data = request.json
    if not update_data:
        return jsonify({"error": "No data provided to update."}), 400

    product = products_collection.find_one({"_id": ObjectId(product_id)})
    if not product:
        return jsonify({"error": "Product not found"}), 404

    result = products_collection.update_one({"_id": ObjectId(product_id)}, {"$set": update_data})
    return jsonify({"message": "Product updated successfully" if result.modified_count == 1 else "No changes made"}), 200

# Route to delete a product
@app.route("/products/<product_id>", methods=["DELETE"])
def delete_product(product_id):
    result = products_collection.delete_one({"_id": ObjectId(product_id)})
    return jsonify({"message": "Product deleted successfully" if result.deleted_count == 1 else "Product not found"}), 200

# Route to serve the index.html
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

if __name__ == "__main__":
    app.run(debug=True)

