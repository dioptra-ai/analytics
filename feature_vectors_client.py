from datetime import datetime
from sqlalchemy import text
from functools import partial

from pymongo import InsertOne

from utils import decode_to_np_array, process_pool_map, thread_pool_submit, mongo_cache
get_cache_collection = mongo_cache.get_cache_collection_factory(collection_name='feature_vectors', collection_size_GiB=10)

from schemas.pgsql import get_sql_engine

sql_engine = get_sql_engine()

MONGODB_UPDATE_BATCH_SIZE = 100

def _fetch_by_ids(ids):
    returned_results = []
    mongodb_requests = []
    def _flush_mongodb_requests(requests):
        if len(requests) > 0:
            try:
                get_cache_collection().bulk_write(requests)
            except Exception as e:
                print(f'WARNING Could not write feature vectors into MongoDB: {e}')

    with sql_engine.connect() as conn:
        results = conn.execution_options(stream_results=True).execute(
            text('''
                SELECT * FROM feature_vectors 
                WHERE id IN :ids
            '''), {
                'ids': tuple(ids)
            }
        )
        
        for row in results:
            update = dict(row)
            update['id'] = str(update['id'])
            mongodb_requests.append(InsertOne(update))
            returned_results.append(update)

            if len(mongodb_requests) >= MONGODB_UPDATE_BATCH_SIZE:
                thread_pool_submit(_flush_mongodb_requests, mongodb_requests)
                mongodb_requests = []

    if len(mongodb_requests) > 0:
        thread_pool_submit(_flush_mongodb_requests, mongodb_requests)

    return returned_results

def get_by_ids(ids, force_fetch=False, flatten=True):
    if len(ids) == 0:
        return []

    tic = datetime.now()
    if force_fetch:
        results = []
    else:
        results = list(get_cache_collection().find({'id': {'$in': ids}}))

    cache_hit_ids = set([r['id'] for r in results])
    cache_miss_ids = set(ids) - cache_hit_ids

    if len(cache_miss_ids):
        results += _fetch_by_ids(list(cache_miss_ids))

    # Reorder results to match the order of ids.
    results = {str(result['id']): result for result in results}
    results = [results[id] for id in ids]

    decoded_vectors = process_pool_map(partial(decode_to_np_array, flatten=flatten), [result['encoded_value'] for result in results])

    print(f'feature vectors cache hit ratio: {len(cache_hit_ids) / len(ids)}. Elapsed time: {datetime.now() - tic}')

    return decoded_vectors

def get_datapoint_ids_for_vector_ids(vector_ids):
    with sql_engine.connect() as conn:
        results = conn.execute(
            text('''
                SELECT predictions.datapoint, feature_vectors.id
                FROM predictions
                JOIN feature_vectors ON feature_vectors.prediction = predictions.id
                WHERE feature_vectors.id IN :ids
            '''), {
                'ids': tuple(vector_ids)
            }
        )

        # Reorder results to match the order of ids.
        results = {str(result['id']): result for result in results}
        results = [results[id]['datapoint'] for id in vector_ids]
        
        return results
