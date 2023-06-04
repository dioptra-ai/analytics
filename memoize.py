# https://gist.github.com/ikuyamada/40f9acd0da2113f9c2d7
import base64
import hashlib
import pickle
from functools import wraps

from utils.mongo_cache import get_cache_collection_factory

class PickleSerializer(object):
    def serialize(self, obj):
        return base64.b64encode(pickle.dumps(obj))

    def deserialize(self, serialized):
        return pickle.loads(base64.b64decode(serialized))

get_cache_collection = get_cache_collection_factory(collection_name='memoized', collection_size_GiB=10)
get_cache_collection().ensure_index('key', unique=True)

def get_memoizer():
    def decorator(func):
        serializer_ins = PickleSerializer()

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            cache_col = get_cache_collection()
            cache_key = hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()
            cached_obj = cache_col.find_one({'key': cache_key})

            print(f'Memoize cache hit: {cached_obj is not None} [{func.__name__}]')

            if cached_obj:

                return serializer_ins.deserialize(cached_obj['result'])
            else:
                result = func(*args, **kwargs)
                cache_col.insert_one({
                    'key': cache_key,
                    'result': serializer_ins.serialize(result)
                })

                return result

        return wrapped_func

    return decorator

memoize = get_memoizer()
