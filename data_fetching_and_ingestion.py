import requests
from pymongo import MongoClient
from pymongo.operations import UpdateOne

# --- CONFIGURATION  ---
PRODUCTS_API_URL = "https://dummyjson.com/products?limit=500"
MONGO_URI = "mongodb://localhost:27017/" 
DATABASE_NAME = "local"
COLLECTION_NAME = "products"

# --- Mongodb connection FUNCTION  ---
def get_mongo_collection():
    """Returns the MongoDB products collection."""
    try:
        client = MongoClient(MONGO_URI) 
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("Connected to MongoDB successfully.")
        return collection
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        return None

# --- INGESTION LOGIC  ---

def ingest_data():
    """
    Fetches product data from the API and stores/upserts it in MongoDB.
    
    Requirements Met:
    - Fetches data from /products endpoint.
    - Handles non-200 responses.
    - Uses upsert by product ID to prevent duplication.
    """
    print("--- Starting data ingestion process ---")
    collection = get_mongo_collection()
    if collection is None:
        print("Aborting ingestion due to MongoDB connection failure.")
        return

    try:
        # Fetch product data from the API
        response = requests.get(PRODUCTS_API_URL)
        
        # Handle basic errors (non-200 responses)
        if response.status_code != 200:
            print(f"API Error: Received status code {response.status_code}. Response: {response.text}")
            return

        data = response.json()
        products = data.get('products', [])
        
        if not products:
            print("API response contained no products to ingest.")
            return

        # Make it easy to re-run ingestion without duplicating data (upsert)
        operations = []
        for product in products:
           operations.append(
                UpdateOne(
                    {'id': product['id']},        
                    {'$set': product},          
                    upsert=True                 
                )
            )
           
        
        # Execute all upsert operations in bulk for efficiency
        result = collection.bulk_write(operations)
        
        print(f"--- Ingestion successful ---")
        print(f"Products Fetched: {len(products)}")
        print(f"Existing Documents Matched/Updated: {result.matched_count}")
        print(f"New Documents Inserted (Upserted): {result.upserted_count}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed due to network or connection error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}")

if __name__ == "__main__":
    ingest_data()
