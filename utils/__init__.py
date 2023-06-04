from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import datetime
from uuid import UUID
import pickle
import simplejson
from io import BytesIO
import base64
from multiprocessing import Pool
import os
from itertools import chain, zip_longest
import lz4.frame
import numpy as np
import pandas as pd

try :
    from cuml.preprocessing import normalize
    import cupy as cp
    cuml_loaded = True
except Exception as e:
    from sklearn.preprocessing import normalize
    cuml_loaded = False

import orjson
from functools import partial
from memoize import memoize


def encode_np_array(np_array):
    if not isinstance(np_array, np.ndarray):
        np_array = np.array(np_array)
    bytes_buffer = BytesIO()
    np.save(bytes_buffer, np_array.astype(dtype=np.float16))

    return base64.b64encode(
        lz4.frame.compress(
            bytes_buffer.getvalue(),
            compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
        )).decode('ascii')

def decode_to_np_array(value, flatten=False, dtype=np.float16):
    result = None
    if isinstance(value, str):
        decoded_bytes = lz4.frame.decompress(base64.b64decode(value))

        if decoded_bytes[:6] == b'\x93NUMPY':
            
            result = np.load(BytesIO(decoded_bytes), allow_pickle=True)

        else:

            result = np.frombuffer(decoded_bytes)
    elif isinstance(value, dict):

        result = np.array(list(value.values()), dtype=np.float32)
    else:
        raise ValueError(f'Unsupported type {type(value)}')
    
    return result.flatten() if flatten else result

def decode_list(value):
    decoded_bytes = lz4.frame.decompress(base64.b64decode(value))
    return orjson.loads(decoded_bytes)

def with_uncompressed_embeddings(data, embeddings_field='embeddings'):
    return [{
        **d,
        embeddings_field: decode_to_np_array(d[embeddings_field])
    } for d in data]

def with_compressed_embeddings(data, embeddings_field='embeddings'):
    return [{
        **d,
        embeddings_field: encode_np_array(d[embeddings_field])
    } for d in data]

def get_uncompressed_embeddings(compressed_dataset: list, embeddings_field: str = 'embeddings') -> list:
    features = process_pool_map(
        partial(post_process_query_rows, np_array_indexes=[embeddings_field]),
        compressed_dataset,
    )
    return [row[0].tolist() for row in features]

# TODO: This function changes the order of the columns in an arbitrary way and needs to be fixed
# and renamed.
def post_process_query_rows(row, non_np_array_indexes=[], json_object_indexes=[], np_array_indexes=[]):
    result = [row[index] for index in non_np_array_indexes]
    result.extend([orjson.loads(row[index]) for index in json_object_indexes])
    result.extend([decode_to_np_array(row[index]) for index in np_array_indexes])
    return result

thread_pool_executor = None
def thread_pool_submit(fn, *args, **kwargs):
    global thread_pool_executor
    if thread_pool_executor is None:
        thread_pool_executor = ThreadPoolExecutor()

    return thread_pool_executor.submit(fn, *args, **kwargs)

# Use this for parallel function that mostly call the network and don't do much computation.
def thread_pool_map(fn, *args):
    global thread_pool_executor
    if thread_pool_executor is None:
        thread_pool_executor = ThreadPoolExecutor()

    return list(thread_pool_executor.map(fn, *args))

# Use this for parallel function that mostly call the network and don't do much computation.
def parallel_threads(funct_list):

    return thread_pool_map(apply_funct, [(f) for f in funct_list])

# Use this for parallel functions that mostly use the CPU and don't call the network.
def process_pool_map(fn, *args):
    with ProcessPoolExecutor() as process_pool_executor:
        # Trying this for very large datasets decoding.
        chunksize = max(1, len(list(args[0])) // os.cpu_count())

        return list(process_pool_executor.map(fn, *args, chunksize=chunksize))

def parallel_processes(funct_list):

    return process_pool_map(apply_funct, [(f) for f in funct_list])

def apply_funct(funct_data):
    if type(funct_data) is tuple:
        funct = funct_data[0]
        data = funct_data[1]
        return funct(data)
    else:
        return funct_data()

class CustomEncoder(simplejson.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ignore_nan=True)

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, pd.Series):
            return obj.to_list()
        return simplejson.JSONEncoder.default(self, obj)

def drop_empties(records):
    '''
        Remove all columns that have only empty values.
    '''
    df = pd.DataFrame(records)
    df.replace('', np.nan, inplace=True)
    df.dropna(axis='columns', how='all', inplace=True)
    df.fillna('', inplace=True)

    return df

def drop_heavy_columns(records):
    '''
        Removed all columns that are weight vectors.
    '''
    df = pd.DataFrame(records)
    df.drop(
        columns=[
            'embeddings', 'original_embeddings', 'features',
            'logits', 'prediction.logits'],
        errors='ignore',
        inplace=True)

    return df

def lean_df(records):
    df = drop_empties(records)
    df = drop_heavy_columns(df)

    return df

def interleave(l1, l2):
    '''
        Interleaves two lists of any length.
    '''
    return [x for x in chain.from_iterable(zip_longest(l1, l2)) if x is not None]

def get_first_n_uniques(iterable, n):
    """Yields (in order) the first n unique elements of iterable. 
    Might yield less if iterable too short."""
    seen = set()
    for e in iterable:
        if e in seen:
            continue

        seen.add(e)

        if len(seen) == n:
            break

    return list(seen)

@memoize
def normalize_vector(vector):
    if cuml_loaded:

        return normalize(cp.array(vector)).get()
    else:

        return normalize(vector)

def compute_iou(bbox_1, bbox_2):

    max_left = max(bbox_1['left'], bbox_2['left'])
    max_top = max(bbox_1['top'], bbox_2['top'])
    min_right = min(bbox_1['left'] + bbox_1['width'], bbox_2['left'] + bbox_2['width'])
    min_bottom = min(bbox_1['top'] + bbox_1['height'], bbox_2['top'] + bbox_2['height'])

    intersection = max(0, min_right - max_left) * max(0, min_bottom - max_top)

    bbox_1_area = bbox_1['width'] * bbox_1['height']
    bbox_2_area = bbox_2['width'] * bbox_2['height']

    iou = intersection / float(bbox_1_area + bbox_2_area - intersection)
    return iou
