import os
from pymongo import MongoClient

def get_mongo_client():
    return MongoClient(os.environ.get("MONGO_URI"))
