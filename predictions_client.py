import pandas as pd

from schemas.pgsql import run_sql_query

def get_predictions_with_metrics(organization_id, datapoint_ids, model_name):
    if len(datapoint_ids) == 0:
        return pd.DataFrame([])

    results = run_sql_query('''
            SELECT
                predictions.id,
                CASE WHEN predictions.task_type = 'INSTANCE_SEGMENTATION' THEN
                    COALESCE(predictions.metrics, '{}') || jsonb_build_object(
                        'entropy', SUM((bboxes.metrics->>'entropy')::float),
                        'variance', SUM((bboxes.metrics->>'variance')::float)
                    )
                    WHEN predictions.task_type = 'LANE_DETECTION' THEN
                    COALESCE(predictions.metrics, '{}') || jsonb_build_object(
                        'entropy', SUM(lane_metrics.entropy),
                        'variance', SUM(lane_metrics.variance)
                    )
                    WHEN predictions.task_type = 'COMPLETION' THEN
                    COALESCE(predictions.metrics, '{}') || jsonb_build_object(
                        'entropy', SUM((completions.metrics->>'entropy')::float),
                        'evaluation_score', AVG((completions.metrics->>'evaluation_score')::float),
                        'rlhf_score', AVG((completions.metrics->>'rlhf_score')::float),
                        'feedback_score', AVG((completions.metrics->>'feedback_score')::float),
                        'hallucination_score', AVG((completions.metrics->>'hallucination_score')::float)
                    )
                ELSE
                    predictions.metrics
                END as metrics,
                predictions.task_type,
                datapoint
            FROM predictions
            LEFT JOIN bboxes ON bboxes.prediction = predictions.id
            LEFT JOIN completions ON completions.prediction = predictions.id
            LEFT JOIN (
                -- Entropies and variances for each lane.
                SELECT SUM((lane_classifications.classification->'metrics'->'entropy')::FLOAT) as entropy,
                    SUM((lane_classifications.classification->'metrics'->'variance')::FLOAT) as variance,
                    lane_classifications.prediction
                FROM (
                    SELECT lanes.prediction, 
                        lanes.id as lane,
                        jsonb_array_elements(lanes.classifications) as classification
                    FROM lanes
                ) AS lane_classifications
                GROUP BY lane_classifications.lane, lane_classifications.prediction
            ) AS lane_metrics ON lane_metrics.prediction = predictions.id
            WHERE predictions.organization_id = :organization_id
                AND datapoint IN :datapoint_ids
                AND predictions.model_name = :model_name
            GROUP BY predictions.id
        ''', params={
            'organization_id': organization_id,
            'datapoint_ids': tuple(datapoint_ids),
            'model_name': model_name or ''
        }
    )

    return pd.DataFrame([{
        'id': result[0],
        'metrics': result[1],
        'task_type': result[2],
        'datapoint': result[3]
    } for result in results])
