"""
Metrics callback module.
"""

import torch
import numpy as np
import pandas as pd
import lightning as L

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
    top_k_accuracy_score
)


class MetricsCallback(L.Callback):
    """Callback to collect predictions during test and compute detailed metrics."""

    def __init__(self, num_classes, class_names):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def reset(self):
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        videos, labels = batch[:2]

        with torch.no_grad():
            model_outputs = pl_module(videos)
            probabilities = torch.softmax(model_outputs, dim=1)
            _, predictions = torch.max(model_outputs, 1)

        self.all_predictions.extend(predictions.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())
        self.all_probabilities.append(probabilities.cpu().numpy())

    def compute_metrics(self):
        """Compute all metrics from collected predictions."""
        if not self.all_probabilities:
            return {}

        all_probabilities = np.concatenate(self.all_probabilities, axis=0)
        all_predictions = np.array(self.all_predictions)
        all_labels = np.array(self.all_labels)

        # Basic accuracy
        accuracy = accuracy_score(all_labels, all_predictions)

        # Precision, Recall, F1-score (macro, weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            all_labels, all_predictions, labels=range(self.num_classes), zero_division=0
        )

        # Top-k accuracy
        top3_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=min(3, self.num_classes))
        top5_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=min(5, self.num_classes))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=range(self.num_classes))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-10)

        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        except:
            roc_auc = None

        # Create DataFrame with per-class metrics
        class_metrics_df = pd.DataFrame({
            'class_id': range(self.num_classes),
            'class_name': [self.class_names.get(i, f'class_{i}') for i in range(self.num_classes)],
            'accuracy': per_class_accuracy,
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1_score': per_class_f1,
            'support': per_class_support,
        })
        class_metrics_df = class_metrics_df.sort_values('f1_score', ascending=False)

        return {
            'accuracy': float(accuracy),
            'top3_accuracy': float(top3_accuracy),
            'top5_accuracy': float(top5_accuracy),
            'roc_auc_ovr': float(roc_auc) if roc_auc is not None else None,
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'per_class_metrics': class_metrics_df.to_dict('records'),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'num_samples_per_class': {self.class_names.get(i, str(i)): int(cm[i].sum())
                                      for i in range(self.num_classes) if cm[i].sum() > 0},
        }