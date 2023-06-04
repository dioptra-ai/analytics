import os
import requests
import pandas as pd

def select_datapoints(organization_id, columns, filters=None, limit=None, offset=None, order_by=None, desc=None, dataset_id=None):
    response = requests.post(f'{os.environ["FRONTEND_URL"]}/api/datapoints/select', headers={
            'x-organization-id': organization_id
        }, json={
        'selectColumns': columns,
        **({'filters': filters} if filters else {}),
        **({'limit': limit} if limit else {}),
        **({'offset': offset} if offset else {}),
        **({'orderBy': order_by} if order_by else {}),
        **({'desc': desc} if desc else {}),
        **({'datasetId': dataset_id} if dataset_id else {}),
    })
    response.raise_for_status()

    return pd.DataFrame(response.json())

def select_predictions(organization_id, columns, filters=None, limit=None, offset=None, order_by=None, desc=None):
    response = requests.post(f'{os.environ["FRONTEND_URL"]}/api/predictions/select', headers={
            'x-organization-id': organization_id
        }, json={
        'selectColumns': columns,
        **({'filters': filters} if filters else {}),
        **({'limit': limit} if limit else {}),
        **({'offset': offset} if offset else {}),
        **({'orderBy': order_by} if order_by else {}),
        **({'desc': desc} if desc else {}),
    })
    response.raise_for_status()

    return pd.DataFrame(response.json())

def select_distinct_model_names(organization_id, filters=None, limit=None, offset=None, order_by=None, desc=None):
    response = requests.post(f'{os.environ["FRONTEND_URL"]}/api/predictions/select-distinct-model-names', headers={
            'x-organization-id': organization_id
        }, json={
        **({'filters': filters} if filters else {}),
        **({'limit': limit} if limit else {}),
        **({'offset': offset} if offset else {}),
        **({'orderBy': order_by} if order_by else {}),
        **({'desc': desc} if desc else {}),
    })
    response.raise_for_status()

    return response.json()

def select_groundtruths(organization_id, columns, filters=None, limit=None, offset=None, order_by=None, desc=None):
    response = requests.post(f'{os.environ["FRONTEND_URL"]}/api/groundtruths/select', headers={
            'x-organization-id': organization_id
        }, json={
        'selectColumns': columns,
        **({'filters': filters} if filters else {}),
        **({'limit': limit} if limit else {}),
        **({'offset': offset} if offset else {}),
        **({'orderBy': order_by} if order_by else {}),
        **({'desc': desc} if desc else {}),
    })
    response.raise_for_status()

    return pd.DataFrame(response.json())

def find_vector_ids_for_datapoints(organization_id, type, filters=None, limit=None, offset=None, order_by=None, desc=None, model_name=None, dataset_id=None):
    response = requests.post(f'{os.environ["FRONTEND_URL"]}/api/datapoints/find-vector-ids', headers={
            'x-organization-id': organization_id
        }, json={
        **({'filters': filters} if filters else {}),
        **({'limit': limit} if limit else {}),
        **({'offset': offset} if offset else {}),
        **({'orderBy': order_by} if order_by else {}),
        **({'desc': desc} if desc else {}),
        **({'modelName': model_name} if model_name else {}),
        **({'datasetId': dataset_id} if dataset_id else {}),
        'type': type,
    })
    response.raise_for_status()

    return response.json()
