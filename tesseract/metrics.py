import Levenshtein as lev
import numpy as np


def evaluate_ocr_batch(
    text_preds, text_gts,
    bbox_preds=None, bbox_gts=None,
    confidences=None
):
    """
    Evaluate OCR results in a batch using multiple metrics:
    - WER (Word Error Rate)
    - CER (Character Error Rate)
    - Bounding Box Regression Error
    - Confidence-Aware WER/CER

    Parameters:
        text_preds (list of str): List of predicted texts.
        text_gts (list of str): List of ground truth texts.
        bbox_preds (list of list or None): List of predicted bounding boxes [[x0, y0, x1, y1], ...].
        bbox_gts (list of list or None): List of ground truth bounding boxes [[x0, y0, x1, y1], ...].
        confidences (list of float or None): List of OCR confidence scores (0 to 1).

    Returns:
        dict: Evaluation results with keys:
            - 'WERs': List of Word Error Rates
            - 'CERs': List of Character Error Rates
            - 'BBoxErrors': List of Bounding box regression errors (if applicable)
            - 'ConfAwareWERs': List of Confidence-weighted WERs (if applicable)
            - 'ConfAwareCERs': List of Confidence-weighted CERs (if applicable)
            - 'AvgWER': Average Word Error Rate
            - 'AvgCER': Average Character Error Rate
            - 'AvgBBoxError': Average Bounding Box Regression Error
            - 'AvgConfAwareWER': Average Confidence-Aware WER
            - 'AvgConfAwareCER': Average Confidence-Aware CER
    """
    assert len(text_preds)==len(text_gts), "different length of words in prediction and GT, probably a word got missed/ split, handaling different length will be implemented soon"

    results = {
        "WERs": [],
        "CERs": [],
        "BBoxErrors": [],
        "ConfAwareWERs": [],
        "ConfAwareCERs": []
    }

    for i in range(len(text_preds)):
        text_pred = text_preds[i]
        text_gt = text_gts[i]
        bbox_pred = bbox_preds[i] if bbox_preds else None
        bbox_gt = bbox_gts[i] if bbox_gts else None
        confidence = confidences[i] if confidences else None

        # WER and CER
        if text_gt and text_pred:
            wer = lev.distance(text_gt.split(), text_pred.split()) / len(text_gt.split()) if text_gt.split() else 1.0
            cer = lev.distance(text_gt, text_pred) / len(text_gt) if text_gt else 1.0
        else:
            wer = None
            cer = None
        results["WERs"].append(wer)
        results["CERs"].append(cer)

        # Bounding Box Regression Error
        if bbox_pred is not None and bbox_gt is not None:
            bbox_pred = np.array(bbox_pred)
            bbox_gt = np.array(bbox_gt)
            bbox_error = np.mean((bbox_pred - bbox_gt) ** 2)
        else:
            bbox_error = None
        results["BBoxErrors"].append(bbox_error)

        # Confidence-Aware WER and CER
        if confidence is not None and confidence >= 0 and text_gt and text_pred:
            conf_aware_wer = confidence/100 * wer if wer is not None else None
            conf_aware_cer = confidence/100 * cer if cer is not None else None
        else:
            conf_aware_wer = None
            conf_aware_cer = None
        results["ConfAwareWERs"].append(conf_aware_wer)
        results["ConfAwareCERs"].append(conf_aware_cer)

    # Average Metrics
    results["AvgWER"] = np.mean([x for x in results["WERs"] if x is not None]) if results["WERs"] else None
    results["AvgCER"] = np.mean([x for x in results["CERs"] if x is not None]) if results["CERs"] else None
    results["AvgBBoxError"] = np.mean([x for x in results["BBoxErrors"] if x is not None]) if results["BBoxErrors"] else None
    results["AvgConfAwareWER"] = np.mean([x for x in results["ConfAwareWERs"] if x is not None]) if results["ConfAwareWERs"] else None
    results["AvgConfAwareCER"] = np.mean([x for x in results["ConfAwareCERs"] if x is not None]) if results["ConfAwareCERs"] else None

    return results
