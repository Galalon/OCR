def match_predictions_to_gt(pred_text,pred_bboxes,gt_text,gt_bboxes, intersect_threshold=0.9):
    matchings = []
    for gt_i in range(len(gt_bboxes)):
        for pred_i in range(len(pred_bboxes)):
            pred_bbox = pred_bboxes[pred_i]
            gt_bbox = gt_bboxes[gt_i]
            intersect = [max(pred_bbox[0],gt_bbox[0]),
                         max(pred_bbox[1],gt_bbox[1]),
                         min(pred_bbox[2], gt_bbox[2]),
                         min(pred_bbox[3], gt_bbox[3]),
                         ]
            bbox_area = lambda x: max((x[2] - x[0]),0)*max((x[3] - x[1]),0)
            intersect_ratio = max(bbox_area(intersect)/bbox_area(pred_bbox),bbox_area(intersect)/bbox_area(gt_bbox))
            if intersect_ratio > intersect_threshold:
                matchings.append((pred_text[pred_i],pred_bbox,gt_text[gt_i],gt_bbox,intersect_ratio))
    return matchings
