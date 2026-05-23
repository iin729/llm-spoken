"""
WSPairs Orality Level estimator g(.) and evaluation utilities.

Public API:
    stat_all_metrics(text)        -> 8-dim oral feature dict
    eval_oscore_and_level(stats)  -> (o_score, level) using the pre-fit
                                     EWM+KMeans model in oral_level_model.pkl
"""

from .stat_metric8 import stat_all_metrics
from .eval_oscore_level import eval_oscore_and_level

__all__ = ["stat_all_metrics", "eval_oscore_and_level"]
