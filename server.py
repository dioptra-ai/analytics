import os
import json
import traceback
import logging

from flask import Flask, Response, request, jsonify
from werkzeug.exceptions import BadRequestKeyError
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from vectors_blueprint import vectors_blueprint

from handlers.distribution import (
    get_groundtruths_distribution, get_predictions_distribution, 
    get_entropy_distribution, get_mislabeling_distribution,
    get_tag_distribution
)

ENVIRONMENT = os.environ['ENVIRONMENT']
COMMIT_REF = os.environ.get('COMMIT_REF')

sentry_sdk.init(
    dsn="https://9f5e1fec08ae4ff08cb392faa0f841af@o1152673.ingest.sentry.io/4504749339115520",
    environment=ENVIRONMENT,
    release=COMMIT_REF,
    integrations=[
        FlaskIntegration(),
    ],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

from utils import CustomEncoder
from exceptions import (ClientException, IllegalArgumentError)
from integrations.storage.cloud_storage_util import CloudStorageUtil
app = Flask(__name__)

app.json_encoder = CustomEncoder
app.register_blueprint(vectors_blueprint, url_prefix='/vectors')

@app.route('/signed-url', methods = ['POST'])
def get_signed_url():
    data = request.json
    organization_id = request.headers.get('x-organization-id')
    url = data['url']

    signed_url = CloudStorageUtil().generate_presigned_file_url(url, organization_id)

    return jsonify(signed_url)


@app.errorhandler(BadRequestKeyError)
def handle_bad_request_key_error(e):
    logging.error(str(e))
    return jsonify({
        'error': {
            'message': 'Missing property: ' + str(e.args)
        }
    }), 400

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    sentry_sdk.capture_exception(error)
    try:
        logging.error(f'Error for {request.path} - body={json.dumps(request.json, indent=4)}')
    except:
        logging.error(f'Error for {request.path} - body={request.data}')
    traceback.print_exc()

    if hasattr(error, 'get_response'):
        status = error.get_response().status_code
    elif hasattr(error, 'status_code'):
        status = error.status_code
    elif isinstance(error, ClientException):
            status = 400
    else:
        status = 500

    response = {
        'success': False,
        'error': {
            'type': error.__class__.__name__,
            'message': str(error),
            'traceback': traceback.format_exc()
        }
    }

    return jsonify(response), status

@app.route('/distribution/groundtruths', methods = ['POST'])
def distribution_groundtruths():
    data = request.json

    result = get_groundtruths_distribution(
        organization_id=data['organization_id'],
        datapoint_filters=data['filters'],
        model_name=data.get('modelName', None),
        dataset_id=data.get('datasetId', None),
    )

    return jsonify(result)

@app.route('/distribution/predictions', methods = ['POST'])
def distribution_predictions():
    data = request.json

    result = get_predictions_distribution(
        organization_id=data['organization_id'],
        datapoint_filters=data['filters'],
        model_name=data['modelName'],
        dataset_id=data.get('datasetId', None),
    )

    return jsonify(result)

@app.route('/distribution/entropy', methods = ['POST'])
def distribution_entropy():
    data = request.json

    result = get_entropy_distribution(
        organization_id=data['organization_id'],
        datapoint_filters=data['filters'],
        model_name=data['modelName'],
        dataset_id=data.get('datasetId', None),
    )

    return jsonify(result)

@app.route('/distribution/mislabeling', methods = ['POST'])
def distribution_mislabeling():
    data = request.json

    result = get_mislabeling_distribution(
        organization_id=data['organization_id'],
        datapoint_filters=data['filters'],
        model_name=data['modelName'],
        dataset_id=data.get('datasetId', None),
    )

    return jsonify(result)

@app.route('/distribution/tag/<tag_name>', methods = ['POST'])
def distribution_tags(tag_name):
    data = request.json

    result = get_tag_distribution(
        organization_id=data['organization_id'],
        datapoint_filters=data['filters'],
        dataset_id=data.get('datasetId', None),
        tag_name=tag_name
    )

    return jsonify(result)