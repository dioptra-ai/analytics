import numpy as np
from flask import Blueprint, jsonify, request

from schemas.pgsql import run_sql_query
import feature_vectors_client
import frontend_client

from dimension_reduction import reduce_dimension, DimensionReductionAlgorithms

vectors_blueprint = Blueprint('vectors_blueprint', __name__)

@vectors_blueprint.route('/reduce-dimensions', methods=['POST'])
def handle_reduce_dimension():
    organization_id = request.headers.get('x-organization-id')
    parameters = request.json
    vector_ids = frontend_client.find_vector_ids_for_datapoints(
        organization_id=organization_id,
        type='EMBEDDINGS',
        filters=parameters['datapoint_filters'],
        model_name=parameters['embeddings_name'], # TODO: Fix this inconsistency. This can include layer names, which isn't a model name.
        dataset_id=parameters['dataset_id'],
        limit=10000 # TODO: Find a way to scale this.
    )
    vectors = run_sql_query('''
        SELECT feature_vectors.id as vector,
            predictions.id as prediction,
            datapoints.id as datapoint
        FROM feature_vectors
        INNER JOIN predictions ON predictions.id = feature_vectors.prediction
        INNER JOIN datapoints ON datapoints.id = predictions.datapoint
        WHERE feature_vectors.id IN :vector_ids
    ''', params={
        'vector_ids': tuple(vector_ids)
    })
    vectors_by_id = {str(vector['vector']): vector for vector in vectors}

    vectors = feature_vectors_client.get_by_ids(vector_ids)

    algorithm = DimensionReductionAlgorithms.UMAP if parameters.get('algorithm_name', 'UMAP') == 'UMAP' else DimensionReductionAlgorithms.TSNE
    reduction = reduce_dimension(vectors, algorithm)

    return jsonify([{
        'x': reduction[i][0],
        'y': reduction[i][1],
        **(vectors_by_id[vector_id])
    } for i, vector_id in enumerate(vector_ids)])
