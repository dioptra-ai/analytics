import numpy as np
import pandas as pd
import math
from collections import Counter
from sqlalchemy import text

import frontend_client
import predictions_client
from schemas.pgsql import get_sql_engine

sql_engine = get_sql_engine()

from exceptions import NotImplementedError
from utils.prediction_groundtruth_pairs import match_predictions_and_groundtruths

def get_groundtruths_distribution(organization_id, filters):

    return _get_distribution_for_labels(organization_id, 'groundtruths', filters)

def get_predictions_distribution(organization_id, filters):
    
    return _get_distribution_for_labels(organization_id, 'predictions', filters)

def _get_distribution_for_labels(organization_id, for_table, filters=[]):

    # Try to find a precomputed distribution for the pred/gt.
    select_labels = frontend_client.select_groundtruths if for_table == 'groundtruths' else frontend_client.select_predictions
    with_distribution_metric = select_labels(
        organization_id=organization_id,
        columns=['metrics.distribution', 'task_type', 'datapoint'],
        filters=filters + [{
            'left': 'metrics.distribution',
            'op': 'IS NOT',
            'right': None
        }]
    )

    if len(with_distribution_metric) > 0:
        # Use precomputed distributions if they exist.
        histogram = {}
        task_type = None
        for _, result in with_distribution_metric.iterrows():
            task_type = result['task_type']
            datapoint = result['datapoint']

            for class_name, value in result['distribution'].items():
                if class_name not in histogram:
                    histogram[class_name] = {
                        'value': value,
                        'datapoints': [datapoint]
                    }
                else:
                    histogram[class_name]['value'] += value
                    histogram[class_name]['datapoints'].append(datapoint)

        return {
            'histogram': histogram,
            'task_type': task_type
        }
    else:
        # Otherwise, compute the distribution.
        labels = select_labels(
            organization_id=organization_id,
            columns=['task_type', 'lanes.id', 'bboxes.class_name', 'class_name', 'datapoint'],
            filters=filters
        )

        if len(labels) == 0:
            return {
                'histogram': {},
                'task_type': None
            }

        task_type = labels['task_type'].unique()
        if len(task_type) > 1:
            raise UserWarning(f'Distribution not available for multiple task types.')
        else:
            task_type = task_type[0]
            if task_type == 'LANE_DETECTION':
                num_lanes_per_label = labels['lanes'].apply(lambda lanes: len(lanes) if lanes is not None else 0).values
                num_bins = max(num_lanes_per_label) + 1
                histogram, bins = np.histogram(num_lanes_per_label, bins=range(num_bins + 1))
                datapoints_by_lane_len = labels.groupby(num_lanes_per_label)['datapoint'].apply(list)

                return {
                    'histogram': {
                        str(bins[i]): {
                            'value': value,
                            'datapoints': datapoints_by_lane_len.get(i, [])
                         } for i, value in enumerate(histogram)
                    },
                    'task_type': task_type,
                    'title': f'Lanes per {for_table}'
                }
            elif task_type == 'CLASSIFICATION':
                counts = Counter(labels['class_name'].values)
                datapoints_by_class = labels.groupby('class_name')['datapoint'].apply(list)

                return {
                    'histogram': {
                        str(name): {
                            'value': value,
                            'datapoints': datapoints_by_class.get(name, [])
                        }
                        for name, value in counts.items()
                    },  
                    'task_type': task_type
                }
            elif task_type == 'INSTANCE_SEGMENTATION' or task_type == 'OBJECT_DETECTION':
                exploded_bboxes = labels.explode('bboxes')
                exploded_bboxes['class_name'] = exploded_bboxes['bboxes'].apply(pd.Series)['class_name']
                num_bboxes_by_bbox_class_name = Counter(exploded_bboxes['class_name'].values)
                datapoints_by_bbox_class_name = exploded_bboxes.groupby('class_name')['datapoint'].apply(list)

                return {
                    'histogram': {
                        str(name): {
                            'value': value,
                            'datapoints': datapoints_by_bbox_class_name.get(name, [])
                        }
                        for name, value in num_bboxes_by_bbox_class_name.items()
                    },
                    'task_type': task_type
                }
            else:
                raise NotImplementedError(f'{for_table} distribution not available for task type {task_type}.')

def get_entropy_distribution(organization_id, datapoint_ids, model_name):
    model_metrics = predictions_client.get_predictions_with_metrics(datapoint_ids=datapoint_ids, organization_id=organization_id, model_name=model_name)

    if len(model_metrics) == 0:
        return {
            'histogram': {},
            'task_type': None
        }

    task_types = model_metrics['task_type'].unique()
    if len(task_types) > 1:
        raise UserWarning(f'Entropy distribution not available for multiple task types.')

    all_entropies = [e if e is not None and not np.isnan(e) else 0 for e in model_metrics['metrics'].apply(lambda m: m['entropy']).values]

    if all_entropies:
        if max(all_entropies) > 1:
            num_bins = math.ceil(max(all_entropies) + 1)
            histogram, bins = np.histogram(all_entropies, bins=range(num_bins + 1))
        else:
            num_bins = 10
            histogram, bins = np.histogram(all_entropies, range=(0, 1), bins=num_bins)

        return {
            'histogram': {
                f'{bins[i]:.2f} - {bins[i+1]:.2f}': {
                    'value': value,
                    'datapoints': model_metrics[(bins[i] <= all_entropies) & (all_entropies < bins[i+1])]['datapoint'].values.tolist()
                } for i, value in enumerate(histogram)
            },
            'task_type': task_types[0]
        }
    else:
        return {
            'histogram': {},
            'task_type': task_types[0]
        }

def get_mislabeling_distribution(organization_id, datapoint_filters, model_name, dataset_id):
    matched_predictions_groundtruths = match_predictions_and_groundtruths(
        organization_id=organization_id,
        model_name=model_name,
        datapoint_filters=datapoint_filters,
        dataset_id=dataset_id,
    )

    task_type = matched_predictions_groundtruths['task_type']

    if task_type == 'LANE_DETECTION':
        lane_pairs = matched_predictions_groundtruths['lane_pairs']
        confidences_per_datapoint = {}
        for pred_lane, _ in lane_pairs:
            confidences_per_datapoint.setdefault(pred_lane['datapoint'], []).append(pred_lane['confidence'])

        mislabeling_score_per_datapoint = {
            datapoint: 1 - min(confidences) if len(confidences) > 0 else 0
            for datapoint, confidences in confidences_per_datapoint.items()
        }
        
        all_mislabeling_scores = list(mislabeling_score_per_datapoint.values())

        if all_mislabeling_scores:
            histogram, bins = np.histogram(all_mislabeling_scores, range=(0, 1), bins=10)

            return {
                'histogram': {
                    f'{bins[i]:.2f} - {bins[i+1]:.2f}': {
                        'value': value,
                        'datapoints': [
                            datapoint for datapoint, mislabeling_score in mislabeling_score_per_datapoint.items()
                            if bins[i] <= mislabeling_score < bins[i+1]
                        ]
                    } for i, value in enumerate(histogram)
                },
                'task_type': task_type
            }
        else:
            return {
                'histogram': {},
                'task_type': task_type
            }
    else:
        raise NotImplementedError(f'Mislabeling is not defined for task type {task_type}')

def get_tag_distribution(organization_id, datapoint_filters, dataset_id, tag_name):
    datapoints = frontend_client.select_datapoints(
        organization_id=organization_id,
        columns=['id'],
        filters=datapoint_filters,
        dataset_id=dataset_id
    )

    if len(datapoints) == 0:
        return {
            'histogram': {}
        }

    with sql_engine.connect() as conn:
        results = conn.execute(
            text(f'''
                SELECT tags.value AS tag_value,
                    ARRAY_AGG(datapoints.id) AS datapoint_ids
                FROM datapoints
                LEFT JOIN tags ON datapoints.id = tags.datapoint
                WHERE datapoints.id IN :datapoint_ids
                    AND tags.name = :tag_name
                GROUP BY tag_value
            '''), {
                'datapoint_ids': tuple(datapoints['id'].values.tolist()),
                'tag_name': tag_name
            }
        )

        histogram = {}
        for row in results:
            histogram[row['tag_value']] = {
                'value': len(row['datapoint_ids']),
                'datapoints': row['datapoint_ids']
            }
        
        return {
            'title': f'{tag_name} distribution',
            'histogram': histogram
        }
