from collections import defaultdict


def merge_non_unique_pairs(pairs):
    # Group by first and second indices
    first_groups = defaultdict(set)
    second_groups = defaultdict(set)

    # Populate the groups
    for a, b in pairs:
        first_groups[a].add(b)
        second_groups[b].add(a)

    # Track visited indices
    visited_first = set()
    visited_second = set()

    result = []

    # Merge groups based on unique first and second indices
    for a, b_list in first_groups.items():
        if a in visited_first:
            continue

        left_indices = {a}
        right_indices = set(b_list)

        # Expand right indices by checking second_groups
        for b in list(right_indices):
            if b not in visited_second:
                left_indices.update(second_groups[b])
                visited_second.add(b)

        visited_first.update(left_indices)

        result.append((sorted(left_indices), sorted(right_indices)))

    return result


def match_predictions_to_gt(pred_text, pred_bboxes, gt_text, gt_bboxes, confidences=None, intersect_threshold=0.9):
    matchings = []
    for gt_i in range(len(gt_bboxes)):
        for pred_i in range(len(pred_bboxes)):
            pred_bbox = pred_bboxes[pred_i]
            gt_bbox = gt_bboxes[gt_i]
            intersect = [max(pred_bbox[0], gt_bbox[0]),
                         max(pred_bbox[1], gt_bbox[1]),
                         min(pred_bbox[2], gt_bbox[2]),
                         min(pred_bbox[3], gt_bbox[3]),
                         ]
            bbox_area = lambda x: max((x[2] - x[0]), 0) * max((x[3] - x[1]), 0)
            intersect_ratio = max(bbox_area(intersect) / bbox_area(pred_bbox),
                                  bbox_area(intersect) / bbox_area(gt_bbox))
            if intersect_ratio > intersect_threshold:
                # matchings.append((pred_text[pred_i],pred_bbox,gt_text[gt_i],gt_bbox,intersect_ratio))
                matchings.append((pred_i, gt_i))
    merged_matchings = merge_non_unique_pairs(matchings)
    eval_stats = {"matched": 0, "merged": 0, "split": 0, "missed": 0, "false_detection": 0}
    new_pred_text = []
    new_pred_bboxes = []
    new_gt_text = []
    new_gt_bboxes = []
    new_confidence = None if confidences is None else []
    merge_bboxes = lambda x: (min([y[0] for y in x]), min([y[1] for y in x]), max([y[2] for y in x]),
                              max([y[3] for y in x]))

    for idxs_pred, idxs_gt in merged_matchings:
        if len(idxs_pred) == 1 and len(idxs_gt) == 1:  # match
            eval_stats["matched"] += 1
            new_pred_text.append(pred_text[idxs_pred[0]])
            new_pred_bboxes.append(pred_bboxes[idxs_pred[0]])
            if confidences is not None:
                new_confidence.append(confidences[idxs_pred[0]])
            new_gt_text.append(gt_text[idxs_gt[0]])
            new_gt_bboxes.append(gt_bboxes[idxs_gt[0]])

        elif len(idxs_pred) > 1 and len(idxs_gt) == 1:  # split
            eval_stats["split"] += 1
            new_pred_text.append(" ".join([pred_text[i] for i in idxs_pred]))
            new_pred_bboxes.append(merge_bboxes([pred_bboxes[i] for i in idxs_pred]))
            if confidences is not None:
                new_confidence.append(sum([confidences[i] for i in idxs_pred]) / len(idxs_pred))  # average
            new_gt_text.append(gt_text[idxs_gt[0]])
            new_gt_bboxes.append(gt_bboxes[idxs_gt[0]])
        elif len(idxs_pred) == 1 and len(idxs_gt) > 1:  # merge
            eval_stats["merged"] += 1
            new_pred_text.append(pred_text[idxs_pred[0]])
            new_pred_bboxes.append(pred_bboxes[idxs_pred[0]])
            if confidences is not None:
                new_confidence.append(confidences[idxs_pred[0]])
            new_gt_text.append("".join([gt_text[i] for i in idxs_gt]))
            new_gt_bboxes.append(merge_bboxes([gt_bboxes[i] for i in idxs_gt]))
        else:
            raise RuntimeError(
                f"cannot resolve matching: gt: {[gt_text[i] for i in idxs_gt]}, pred {[pred_text[i] for i in idxs_pred]}")

    missed_gts = list(set(range(len(gt_text))) - set(sum([t[1] for t in merged_matchings], [])))
    eval_stats["missed"] = len(missed_gts)
    for gt_idx in missed_gts:
        new_pred_text.append(None)
        new_pred_bboxes.append(None)
        if confidences is not None:
            new_confidence.append(None)
        new_gt_text.append(gt_text[gt_idx])
        new_gt_bboxes.append(gt_bboxes[gt_idx])

    false_detections = list(set(range(len(pred_text))) - set(sum([t[0] for t in merged_matchings], [])))
    eval_stats["false_detection"] = len(false_detections)
    for pred_idx in false_detections:
        new_pred_text.append(pred_text[pred_idx])
        new_pred_bboxes.append(pred_bboxes[pred_idx])
        if confidences is not None:
            new_confidence.append(confidences[pred_idx])
        new_gt_text.append(None)
        new_gt_bboxes.append(None)
    return new_pred_text, new_pred_bboxes, new_gt_text, new_gt_bboxes,new_confidence, eval_stats


if __name__ == "__main__":

    text_a = ["Perfect match", "Slight mismatch", "Split Match", "Non existent", "MergedM", "atch"]
    text_b = ["Perfect match", "Slight mismatch", "SplitM", " atch", "Merged Match"]
    bboxes_a = [
        (10, 20, 50, 60),  # Perfect match
        (60, 80, 120, 160),  # Slight mismatch
        (130, 150, 200, 220),  # Split in List B
        (300, 350, 400, 450),  # Only in A (missing in B)
        (500, 550, 550, 650),  # Merged in List B first part
        (557, 550, 600, 650)  # Merged in List B second part part
    ]
    bboxes_b = [
        (10, 20, 50, 60),  # Perfect match
        (58, 78, 122, 162),  # Slight mismatch
        (130, 150, 165, 220),  # First part of the split box
        (170, 150, 200, 220),  # Second part of the split box
        (495, 545, 605, 655)  # Merged version of two boxes in A
    ]
    confidences = [a / 10 for a in range(9, 3,-1)]

    new_text_a, new_bboxes_a, new_text_b, new_bboxes_b,new_confidence, stats = match_predictions_to_gt(text_a, bboxes_a, text_b, bboxes_b,
                                                                                        confidences=confidences,
                                                                                        intersect_threshold=0.9)
    print(f"stats: {stats}")
    print(f"text a: {new_text_a}")
    print(f"text b: {new_text_b}")
    print(f"bbox a: {new_bboxes_a}")
    print(f"bbox a: {new_bboxes_b}")
    print(f"confidence: {new_confidence}")
