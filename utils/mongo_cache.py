# https://gist.github.com/ikuyamada/40f9acd0da2113f9c2d7
import os
from pymongo import MongoClient

GiB = 1024 * 1024 * 1024

def get_mongo_client():
    return MongoClient(os.environ.get('MONGO_CACHE_URI'), replicaSet=os.environ.get('MONGO_CACHE_REPLICASET', None), readPreference='nearest', localThresholdMS=0)

def get_cache_collection_factory(collection_name, collection_size_GiB):
    def get_cache_collection():
        return get_mongo_client().dioptra_cache[collection_name]

    try:
        mongo_client = get_mongo_client()
        try:
            mongo_client.dioptra_cache.create_collection(collection_name, capped=True, size=collection_size_GiB * GiB)
        except Exception as e:
            print(f'INFO: creating collection "{collection_name}": {e}. Running collMod instead.')
            mongo_client.dioptra_cache.command('collMod', collection_name, cappedSize=collection_size_GiB * GiB)
    except Exception as e:
        print(f'WARNING: Error initializing cache collection {collection_name}: {e}')

    return get_cache_collection
