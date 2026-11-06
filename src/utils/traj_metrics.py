from itertools import chain
from dataclasses import dataclass
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class PairwiseMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        d = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / d if d > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

    @property
    def mcc(self) -> float:
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        num = tp * tn - fp * fn
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return num / den if den > 0 else 0.0

    @property
    def jaccard(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def fowlkes_mallows(self) -> float:
        p, r = self.precision, self.recall
        return (p * r) ** 0.5

    def __add__(self, other: "PairwiseMetrics") -> "PairwiseMetrics":
        return PairwiseMetrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)


def labels_from_groups(groups: dict) -> tuple[list[int], list[int]]:
    """Convert {gid: [tids]} to (tids_list, labels_list)."""
    tids = sorted(set(chain.from_iterable(groups.values())))
    gid_of = {tid: gid for gid, members in groups.items() for tid in members}
    return tids, [gid_of[tid] for tid in tids]


def calculate_ari_nmi(groups_gt: dict, groups_pred: dict) -> tuple[float, float]:
    tids_gt, labels_gt = labels_from_groups(groups_gt)
    tids_pred, labels_pred = labels_from_groups(groups_pred)

    common = sorted(set(tids_gt) & set(tids_pred))
    gt_map = dict(zip(tids_gt, labels_gt))
    pred_map = dict(zip(tids_pred, labels_pred))
    y_true = [gt_map[t] for t in common]
    y_pred = [pred_map[t] for t in common]

    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
    )


def calculate_pairwise_metrics(groups_gt: dict, groups_pred: dict) -> PairwiseMetrics:
    """
    Pairwise TP/FP/FN/TN between two groupings.

    :param groups_gt: ground-truth groups (gid -> [tids])
    :param groups_pred: predicted groups (gid -> [tids])
    :return: PairwiseMetrics with tp, fp, fn, tn
    """
    tids = sorted(
        set(chain.from_iterable(groups_gt.values())) |
        set(chain.from_iterable(groups_pred.values()))
    )
    n = len(tids)

    def pair_set(groups):
        s = set()
        for members in groups.values():
            m = sorted(members)
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    s.add((m[i], m[j]))
        return s

    gt_pairs = pair_set(groups_gt)
    pred_pairs = pair_set(groups_pred)

    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    tn = n * (n - 1) // 2 - tp - fp - fn

    return PairwiseMetrics(tp=tp, fp=fp, fn=fn, tn=tn)