import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def evaluate_scores(rows, thresholds, id_list):
    metrics, rocs = {}, {}
    for machine in id_list:
        subset = [row for row in rows if row['machine_id'] == machine]
        labels = np.array([row['label'] for row in subset])
        scores = np.array([row['score'] for row in subset])
        fpr, tpr, cuts = roc_curve(labels, scores)
        metrics[machine] = dict(auc=float(roc_auc_score(labels, scores)),
            pauc=float(roc_auc_score(labels, scores, max_fpr=.1)),
            confusion=confusion_matrix(labels, scores > thresholds[machine], labels=[0, 1]).tolist(),
            threshold=thresholds[machine], test_clips=len(subset))
        rocs[machine] = dict(fpr=fpr.tolist(), tpr=tpr.tolist(),
                            thresholds=[float(value) if np.isfinite(value) else None for value in cuts])
    metrics['mean_auc'] = float(np.mean([metrics[key]['auc'] for key in id_list]))
    metrics['mean_pauc'] = float(np.mean([metrics[key]['pauc'] for key in id_list]))
    metrics['pauc_definition'] = 'sklearn standardized partial ROC AUC, max_fpr=0.1'
    return metrics, rocs
