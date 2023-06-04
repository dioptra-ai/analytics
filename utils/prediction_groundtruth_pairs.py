# See for LANE_DETECTION: https://github.com/Turoad/CLRNet/blob/main/clrnet/utils/culane_metric.py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cp = np
    cupy_available = False

import cv2

from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment

import frontend_client
from utils import decode_list

def _generate_segmentation_pairs(prediction, groundtruth):
    # There should be only one groundtruth and prediction per datapoint for this task_type so take the first one.
    if prediction and 'encoded_resized_segmentation_class_mask' in prediction:
        prediction['encoded_resized_segmentation_class_mask'] = decode_list(prediction['encoded_resized_segmentation_class_mask'])
    if groundtruth and 'encoded_resized_segmentation_class_mask' in groundtruth:
        groundtruth['encoded_resized_segmentation_class_mask'] = decode_list(groundtruth['encoded_resized_segmentation_class_mask']) 

    # TODO: Remove this check when ingestion ensures the condition.
    if prediction is not None and groundtruth is not None:
        pred_mask = np.array(prediction.get('encoded_resized_segmentation_class_mask', []))
        gt_mask = np.array(groundtruth.get('encoded_resized_segmentation_class_mask', []))

        if pred_mask.shape != gt_mask.shape:
            print(f'WARNING: Prediction and groundtruth masks have different shapes for datapoint {prediction.get("datapoint", None)}. Prediction shape: {pred_mask.shape}, groundtruth shape: {gt_mask.shape}')
            return None
    
    return (prediction, groundtruth)

def culane_metric(pred, anno, iou_thresholds):
    def draw_lane(lane, img_shape, thickness):
        # opencv doesn't take cupy arrays.
        img = np.zeros(img_shape, dtype=np.uint8)
        lane = lane.astype(np.int32)
        for p1, p2 in zip(lane[:-1], lane[1:]):
            cv2.line(img,
                    tuple(p1),
                    tuple(p2),
                    color=(255, 255, 255),
                    thickness=thickness)

        return img

    def discrete_cross_iou(xs, ys):
        IMAGE_SIZE_CAP = 256
        # img_shape = (img_width, img_height) is the shape of the image that the lanes are drawn on.
        # xs and ys are arrays of shape (num_lanes, num_points, 2) where the last dimension is a point.
        # First, downsample all points to a maximum resolution of IMAGE_SIZE_CAP X IMAGE_SIZE_CAP.
        # Then, draw the lanes on a black image of shape img_shape.
        # Finally, compute the intersection and union of the two images.
        max_of_rows = int(max([max(x[:, 0]) for x in xs] + [max(y[:, 0]) for y in ys]))
        max_of_cols = int(max([max(x[:, 1]) for x in xs] + [max(y[:, 1]) for y in ys]))
        downsampled_width = min(IMAGE_SIZE_CAP, max_of_cols)
        downsampled_height = downsampled_width * max_of_rows // max_of_cols
        downsample_ratio = downsampled_width / max_of_cols

        # Downsample the points.
        xs = [np.array([(int(x * downsample_ratio), int(y * downsample_ratio)) for x, y in lane]) for lane in xs]
        ys = [np.array([(int(x * downsample_ratio), int(y * downsample_ratio)) for x, y in lane]) for lane in ys]

        img_shape = (downsampled_width + 1, downsampled_height + 1)

        # 5% seems like a good hypothesis for lane thickness: https://github.com/Turoad/CLRNet/blob/main/clrnet/utils/culane_metric.py#L27
        # To mitigate the effects of downsampling on iou correctness, take avantage of antialising and use xs * (broadcasted_xs & ys) instead of xs.
        lane_thickness = int(downsampled_width * 0.05)
        xs = cp.array([draw_lane(lane, img_shape=img_shape, thickness=lane_thickness) > 0 for lane in xs])
        ys = cp.array([draw_lane(lane, img_shape=img_shape, thickness=lane_thickness) > 0 for lane in ys])

        broadcasted_xs = xs[:, cp.newaxis]
        intersections = (broadcasted_xs & ys).sum(axis=(2, 3))
        unions = (broadcasted_xs | ys).sum(axis=(2, 3))

        ious = intersections / (unions + 10e-36)

        return ious.get() if cupy_available else ious

    def interp(points, n=50):
        needs_dedup = True
        while needs_dedup:
            deduped_points = [p for i, p in enumerate(points) if p != points[i - 1] or i == 0]
            needs_dedup = len(deduped_points) != len(points)
            points = deduped_points

        x = [x for x, _ in points]
        y = [y for _, y in points]
        tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

        u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)

        return np.array(splev(u, tck)).T

    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]
    
    if len(pred) == 0 or len(anno) == 0:

        return _metric, []

    interp_pred = [interp(pred_lane, n=5) for pred_lane in pred if len(pred_lane) > 1]  # (4, 50, 2)

    interp_anno = [interp(anno_lane, n=5) for anno_lane in anno if len(anno_lane) > 1]  # (4, 50, 2)

    ious = discrete_cross_iou(interp_pred, interp_anno)
    
    pred_ind, gt_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[pred_ind, gt_ind] > thr).sum())    
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]

    return _metric, list(zip(pred_ind, gt_ind))

def get_lane_datection_datapoint_metric(datapoint_id, predictions, groundtruths):
    flat_prediction_lanes = [lane for prediction in predictions for lane in prediction['lanes']]
    flat_prediction_coco_polylines = [lane['coco_polyline'] for lane in flat_prediction_lanes]
    predictions_polylines = [list(zip(polyline[:-1:2], polyline[1::2])) for polyline in flat_prediction_coco_polylines]

    flat_groundtruth_lanes = [lane for groundtruth in groundtruths for lane in groundtruth['lanes']]
    flat_groundtruth_coco_polylines = [lane['coco_polyline'] for lane in flat_groundtruth_lanes]
    groundtruths_polylines = [list(zip(polyline[:-1:2], polyline[1::2])) for polyline in flat_groundtruth_coco_polylines]

    tp_fp_fn, pred_gt_pairs = culane_metric(predictions_polylines, groundtruths_polylines, iou_thresholds=[0.5])

    return {
        'tp_fp_fn': tp_fp_fn,
        'lane_pairs': [({
            'datapoint': datapoint_id,
            **flat_prediction_lanes[pred_ind],
        }, {
            'datapoint': datapoint_id,
            **flat_groundtruth_lanes[gt_ind],
        }) for pred_ind, gt_ind in pred_gt_pairs]
    }

def match_predictions_and_groundtruths(organization_id, model_name, datapoint_filters=[], dataset_id=None, datapoint_offset=None, datapoint_limit=None):
    datapoints = frontend_client.select_datapoints(
        organization_id=organization_id, 
        columns=['id', 'metadata'], 
        filters=(datapoint_filters or []) + [{
            'left': 'predictions.model_name',
            'op': '=',
            'right': model_name
        }], 
        dataset_id=dataset_id, 
        offset=datapoint_offset, 
        limit=datapoint_limit
    )

    with ThreadPoolExecutor() as executor:
        groundtruths = executor.submit(frontend_client.select_groundtruths,
            organization_id=organization_id, 
            columns=[
                'datapoint', 'task_type', 'id',
                'encoded_resized_segmentation_class_mask', 'class_name', 'class_names',
                'lanes.coco_polyline', 'lanes.id', 'lanes.confidence', 'lanes.prediction', 'lanes.groundtruth'
            ],
            filters=[{
                'left': 'datapoint',
                'op': 'in',
                'right': datapoints['id'].tolist()
            }]
        )
        predictions = executor.submit(frontend_client.select_predictions,
            organization_id=organization_id, 
            columns=[
                'datapoint', 'task_type', 'id',
                'encoded_resized_segmentation_class_mask', 'class_name', 'class_names',
                'lanes.coco_polyline', 'lanes.id', 'lanes.confidence', 'lanes.prediction', 'lanes.groundtruth'
            ],
            filters=[{
                'left': 'datapoint',
                'op': 'in',
                'right': datapoints['id'].tolist()
            }, {
                'left': 'model_name',
                'op': '=',
                'right': model_name
            }]
        )

        groundtruths = groundtruths.result().to_dict('records')
        predictions = predictions.result().to_dict('records')
        task_type = predictions[0]['task_type']

        groundtruths_per_datapoint = {}
        for groundtruth in groundtruths:
            if groundtruth['datapoint'] not in groundtruths_per_datapoint:
                groundtruths_per_datapoint[groundtruth['datapoint']] = []
            groundtruths_per_datapoint[groundtruth['datapoint']].append(groundtruth)

        predictions_per_datapoint = {}
        for prediction in predictions:
            if prediction['datapoint'] not in predictions_per_datapoint:
                predictions_per_datapoint[prediction['datapoint']] = []
            predictions_per_datapoint[prediction['datapoint']].append(prediction)

        if task_type == 'SEMANTIC_SEGMENTATION':
            pred_gt_pairs = []
            with ProcessPoolExecutor() as executor:
                for datapoint_id in set(groundtruths_per_datapoint.keys()):
                    # There should be at most one of each for this task type.
                    prediction = None
                    if datapoint_id in predictions_per_datapoint and len(predictions_per_datapoint[datapoint_id]) > 0:
                        prediction = predictions_per_datapoint[datapoint_id][0]

                    groundtruth = None
                    if datapoint_id in groundtruths_per_datapoint and len(groundtruths_per_datapoint[datapoint_id]) > 0:
                        groundtruth = groundtruths_per_datapoint[datapoint_id][0]

                    pred_gt_pairs.append(executor.submit(_generate_segmentation_pairs, prediction, groundtruth))
                
                groundtruths = []
                groundtruths_class_names = {}
                predictions = []
                prediction_groundtruth_pairs = [pair.result() for pair in pred_gt_pairs if pair.result() is not None]
                for prediction, groundtruth in prediction_groundtruth_pairs:
                    predictions.append(prediction.get('encoded_resized_segmentation_class_mask', None) if prediction is not None else None)
                    groundtruths.append(groundtruth.get('encoded_resized_segmentation_class_mask', None) if groundtruth is not None else None)
                    for i, class_name in enumerate(groundtruth.get('class_names', [])):
                        groundtruths_class_names[i] = class_name

                return {
                    'pairs': zip(predictions, groundtruths),
                    'task_type': task_type,
                    'class_names': groundtruths_class_names
                }
        if task_type == 'LANE_DETECTION':
            # Calculate TP, FP, FN for each datapoint.\
            tp_fp_fn = {}
            lane_pairs = []
            results_per_datapoint_id = {}

            if cupy_available:
                executor = ThreadPoolExecutor()
            else:
                executor = ProcessPoolExecutor()

            try:
                for datapoint_id in groundtruths_per_datapoint.keys():
                    predictions = predictions_per_datapoint[datapoint_id]
                    groundtruths = groundtruths_per_datapoint[datapoint_id]

                    results_per_datapoint_id[datapoint_id] = executor.submit(get_lane_datection_datapoint_metric, datapoint_id, predictions, groundtruths)
            finally:
                executor.shutdown(wait=False)

            for datapoint_id, future in results_per_datapoint_id.items():
                result = future.result()
                tp_fp_fn[datapoint_id] = result['tp_fp_fn']
                lane_pairs.extend(result['lane_pairs'])
            
            tp = sum([tp_fp_fn[datapoint_id][0.5][0] for datapoint_id in tp_fp_fn.keys()])
            fp = sum([tp_fp_fn[datapoint_id][0.5][1] for datapoint_id in tp_fp_fn.keys()])
            fn = sum([tp_fp_fn[datapoint_id][0.5][2] for datapoint_id in tp_fp_fn.keys()])

            return {
                'tp_fp_fn': [tp, fp, fn],
                'lane_pairs': lane_pairs,
                'task_type': task_type
            }
        else:
            raise Exception(f'Not available for task type {task_type}')
