import numpy as np
import pandas as pd
from collections import Counter
from sqlalchemy import text

import frontend_client
import predictions_client
from schemas.pgsql import get_sql_engine

sql_engine = get_sql_engine()

from exceptions import NotImplementedError
from utils.prediction_groundtruth_pairs import match_predictions_and_groundtruths

def get_groundtruths_distribution(organization_id, datapoint_filters, model_name=None, dataset_id=None):

    return _get_distribution_for_labels(organization_id, 'groundtruths', datapoint_filters, model_name, dataset_id)

def get_predictions_distribution(organization_id, datapoint_filters, model_name=None, dataset_id=None):
    
    return _get_distribution_for_labels(organization_id, 'predictions', datapoint_filters, model_name, dataset_id)

def _get_distribution_for_labels(organization_id, for_table, datapoint_filters=[], model_name=None, dataset_id=None):
    datapoints = frontend_client.select_datapoints(
        organization_id=organization_id,
        columns=['id'],
        filters=datapoint_filters + ([{
            'left': 'predictions.model_name',
            'op': '=',
            'right': model_name
        }] if model_name else []),
        dataset_id=dataset_id
    )

    if len(datapoints) == 0:
        return {
            'histogram': {},
            'task_type': None
        }

    # select_labels = frontend_client.select_groundtruths if for_table == 'groundtruths' else frontend_client.select_predictions
    labels = frontend_client.select_groundtruths(
        organization_id=organization_id,
        columns=['task_type', 'lanes.id', 'bboxes.class_name', 'completions.text', 'class_name', 'datapoint'],
        filters=[{
            'left': 'datapoint',
            'op': 'in',
            'right': datapoints['id'].values.tolist()
        }],
    ) if for_table == 'groundtruths' else frontend_client.select_predictions(
        organization_id=organization_id,
        columns=['task_type', 'lanes.id', 'bboxes.class_name', 'completions.text', 'class_name', 'datapoint'],
        filters=[{
            'left': 'datapoint',
            'op': 'in',
            'right': datapoints['id'].values.tolist()
        }] + ([{
            'left': 'model_name',
            'op': '=',
            'right': model_name
        }] if model_name else []),
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
                        'datapoints': datapoints_by_lane_len.get(i, []),
                        'index': i
                        } for i, _ in enumerate(histogram)
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
                        'datapoints': datapoints_by_class.get(name, [])
                    }
                    for name, _ in counts.items()
                },  
                'task_type': task_type
            }
        elif task_type == 'INSTANCE_SEGMENTATION' or task_type == 'OBJECT_DETECTION':
            exploded_bboxes = labels.explode('bboxes')
            if not 'class_name' in exploded_bboxes['bboxes'].apply(pd.Series):
                return {
                    'histogram': {},
                    'task_type': task_type
                }

            exploded_bboxes['class_name'] = exploded_bboxes['bboxes'].apply(pd.Series)['class_name']
            # Remove rows that have nan class_name
            exploded_bboxes = exploded_bboxes[~exploded_bboxes['class_name'].isna()]
            num_bboxes_by_bbox_class_name = Counter(c for c in exploded_bboxes['class_name'].to_list() if c)
            sorted_num_bboxes_by_bbox_class_name = sorted(num_bboxes_by_bbox_class_name.items(), key=lambda x: x[0], reverse=False)
            datapoints_by_bbox_class_name = exploded_bboxes.groupby('class_name')['datapoint'].apply(list)

            return {
                'histogram': {
                    str(name): {
                        'datapoints': datapoints_by_bbox_class_name.get(name, [])
                    }
                    for i, (name, _) in enumerate(sorted_num_bboxes_by_bbox_class_name)
                },
                'task_type': task_type
            }
        elif task_type == 'COMPLETION':
            exploded_completions = labels.explode('completions')
            if not 'text' in exploded_completions['completions'].apply(pd.Series):
                return {
                    'histogram': {},
                    'task_type': task_type
                }
            
            exploded_completions['text'] = exploded_completions['completions'].apply(pd.Series)['text']
            exploded_completions['token_length'] = exploded_completions['text'].apply(lambda t: len(t.split(' ')))
            # Remove rows that have nan text
            exploded_completions = exploded_completions[~exploded_completions['text'].isna()]
            token_length_by_completion_text = Counter(len(t.split(' ')) for t in exploded_completions['text'].to_list() if t)
            sorted_token_length_by_completion_text = sorted(token_length_by_completion_text.items(), reverse=False)
            datapoints_by_completion_text = exploded_completions.groupby('token_length')['datapoint'].apply(list)

            return {
                'histogram': {
                    str(name): {
                        'datapoints': datapoints_by_completion_text.get(name, [])
                    }
                    for i, (name, _) in enumerate(sorted_token_length_by_completion_text)
                },
                'task_type': task_type
            }
        else:
            raise NotImplementedError(f'{for_table} distribution not available for task type {task_type}.')

def get_metric_distribution(organization_id, metric, datapoint_filters, model_name, dataset_id):
    datapoints = frontend_client.select_datapoints(
        organization_id=organization_id,
        columns=['id'],
        filters=datapoint_filters,
        dataset_id=dataset_id
    )
    model_metrics = predictions_client.get_predictions_with_metrics(organization_id=organization_id, model_name=model_name, datapoint_ids=datapoints['id'].values.tolist())

    if len(model_metrics) == 0:
        return {
            'histogram': {},
            'task_type': None
        }

    task_types = model_metrics['task_type'].unique()
    if len(task_types) > 1:
        raise UserWarning(f'Entropy distribution not available for multiple task types.')

    metric_values = [e if e is not None and not np.isnan(e) else 0 for e in model_metrics['metrics'].apply(lambda m: m.get(metric, None)).values]

    if metric_values:
        histogram, bins = np.histogram(metric_values, bins=10)

        return {
            'histogram': {
                f'{bins[i]:.2f} - {bins[i+1]:.2f}': {
                    'datapoints': model_metrics[(bins[i] <= metric_values) & (metric_values < bins[i+1])]['datapoint'].values.tolist(),
                    'index': i
                } for i, _ in enumerate(histogram)
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
                        'datapoints': [
                            datapoint for datapoint, mislabeling_score in mislabeling_score_per_datapoint.items()
                            if bins[i] <= mislabeling_score < bins[i+1]
                        ],
                        'index': i
                    } for i, _ in enumerate(histogram)
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

        return {
            'title': f'{tag_name} distribution',
            'histogram': {
                row['tag_value']: {
                    'datapoints': row['datapoint_ids']
                } for i, row in enumerate(results)
            }
        }
