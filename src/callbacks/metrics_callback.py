"""
Metrics callback module.
"""
import torch
import numpy as np
import pandas as pd
import lightning as L

from typing import Any, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
    top_k_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)


class MetricsCallback(L.Callback):
    """Callback to collect predictions during test and compute detailed metrics."""

    def __init__(self, num_classes, class_names):
        """
        Callback initialization.

        :param num_classes: Number of classes.
        :param class_names: Class names.
        """
        super().__init__()
        self._num_classes = num_classes
        self._class_names = class_names
        self._all_classifier_preds = []
        self._all_classifier_labels = []
        self._all_classifier_probs = []
        self._all_severity_preds = []
        self._all_levels = []

    def reset(self):
        """Resets callback."""
        self._all_classifier_preds = []
        self._all_classifier_labels = []
        self._all_classifier_probs = []
        self._all_severity_preds = []
        self._all_levels = []

    def on_test_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        """
        Collect model predictions and true labels for metrics computation.

        :param trainer: Lightning Trainer instance.
        :param pl_module: LightningModule being evaluated.
        :param outputs: Outputs from test_step (unused, predictions recomputed here).
        :param batch: Batch tuple (videos, labels, levels).
        :param batch_idx: Index of the batch.
        :param dataloader_idx: Index of the dataloader.
        """
        videos, labels, levels = batch[:3]

        with torch.no_grad():
            model_outputs = pl_module(videos)

            if isinstance(model_outputs, tuple):
                logits1, logits2 = model_outputs
                # Head 1 (Severity)
                severity_preds = torch.sigmoid(logits1.squeeze(-1))
                self._all_severity_preds.extend(severity_preds.cpu().tolist())
                self._all_levels.extend(levels.cpu().tolist())
            else:
                logits2 = model_outputs

            # Head 2 (Classifier)
            mask = labels != -1
            if mask.any():
                probabilities = torch.softmax(logits2[mask], dim=1)
                _, predictions = torch.max(logits2[mask], 1)
                self._all_classifier_preds.extend(predictions.cpu().tolist())
                self._all_classifier_labels.extend(labels[mask].cpu().tolist())
                self._all_classifier_probs.append(probabilities.cpu().numpy())

    def compute_metrics(self):
        """Compute all metrics from collected predictions."""
        if not self._all_classifier_probs:
            return {}

        all_probabilities = np.concatenate(self._all_classifier_probs, axis=0)
        all_predictions = np.array(self._all_classifier_preds)
        all_labels = np.array(self._all_classifier_labels)

        # ==================== HEAD 2 (CLASSIFIER) ====================
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
            all_labels, all_predictions, labels=range(self._num_classes), zero_division=0
        )

        # Top-k accuracy
        top3_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=min(3, self._num_classes))
        top5_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=min(5, self._num_classes))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=range(self._num_classes))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-10)

        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = None

        # Create DataFrame with per-class metrics
        class_metrics_df = pd.DataFrame({
            'class_id': range(self._num_classes),
            'class_name': [self._class_names.get(i, f'class_{i}') for i in range(self._num_classes)],
            'accuracy': per_class_accuracy,
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1_score': per_class_f1,
            'support': per_class_support,
        })
        class_metrics_df = class_metrics_df.sort_values('f1_score', ascending=False)

        if not self._all_severity_preds:
            metrics = {
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
                'num_samples_per_class': {self._class_names.get(i, str(i)): int(cm[i].sum())
                                          for i in range(self._num_classes) if cm[i].sum() > 0},
            }
        else:
            metrics = {
                'head2_accuracy': float(accuracy),
                'head2_top3_accuracy': float(top3_accuracy),
                'head2_top5_accuracy': float(top5_accuracy),
                'head2_roc_auc_ovr': float(roc_auc) if roc_auc is not None else None,
                'head2_precision_macro': float(precision_macro),
                'head2_recall_macro': float(recall_macro),
                'head2_f1_macro': float(f1_macro),
                'head2_precision_weighted': float(precision_weighted),
                'head2_recall_weighted': float(recall_weighted),
                'head2_f1_weighted': float(f1_weighted),
                'head2_per_class_metrics': class_metrics_df.to_dict('records'),
                'head2_confusion_matrix': cm.tolist(),
                'head2_confusion_matrix_normalized': cm_normalized.tolist(),
                'head2_num_samples_per_class': {self._class_names.get(i, str(i)): int(cm[i].sum())
                                          for i in range(self._num_classes) if cm[i].sum() > 0},
            }

        # ==================== HEAD 1 (SEVERITY) ====================
        if self._all_severity_preds:
            severity_preds = np.array(self._all_severity_preds)
            levels = np.array(self._all_levels)

            # MAE
            mae = mean_absolute_error(levels, severity_preds)

            # R^2
            r2 = r2_score(levels, severity_preds)

            # RMSE
            rmse = np.sqrt(mean_squared_error(levels, severity_preds))

            # Pearson correlation
            try:
                corr, _ = pearsonr(levels, severity_preds)
            except ValueError:
                corr = None

            # Confusion matrix by levels (0.0 / 0.4 / 0.7 / 1.0)
            levels_unique = np.array([0.0, 0.4, 0.7, 1.0])

            def snap(pred):
                return levels_unique[np.argmin(np.abs(levels_unique - pred))]

            snapped_preds = np.array([snap(p) for p in severity_preds])
            snapped_levels = np.array([snap(l) for l in levels])

            level_cm = confusion_matrix(snapped_levels, snapped_preds, labels=levels_unique)
            level_cm_normalized = level_cm.astype('float') / (level_cm.sum(axis=1)[:, np.newaxis] + 1e-10)

            # Level accuracy
            level_accuracy = accuracy_score(snapped_levels, snapped_preds)

            metrics.update({
                'head1_mae': float(mae),
                'head1_rmse': float(rmse),
                'head1_r2': float(r2),
                'head1_pearson_corr': float(corr) if corr is not None else None,
                'head1_level_accuracy': float(level_accuracy),
                'head1_level_confusion_matrix': level_cm.tolist(),
                'head1_level_confusion_matrix_normalized': level_cm_normalized.tolist(),
                'head1_level_labels': levels_unique.tolist(),
            })

        return metrics