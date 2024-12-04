"""Detector base class.

    class name: BaseDetector

    Adapted from the BaseDetector class in sktime.

Scitype defining methods:
    fitting                         - fit(self, X, y=None)
    detecting, sparse format        - predict(self, X)
    detecting, dense format         - transform(self, X)
    detection scores, dense         - transform_scores(self, X)  [optional]
    updating (temporal)             - update(self, X, y=None)  [optional]

Each detector type (e.g. point anomaly detector, collective anomaly detector,
changepoint detector) are subclasses of BaseDetector (task tag in sktime).
A detector type is defined by the content and format of the output of the predict
method. Each detector type therefore has the following methods for converting between
sparse and dense output formats:
    converting sparse output to dense - sparse_to_dense(y_sparse, index, columns)
    converting dense output to sparse - dense_to_sparse(y_dense)  [optional]

Convenience methods:
    update&detect   - update_predict(self, X)
    fit&detect      - fit_predict(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()

Needs to be implemented for a concrete detector:
    _fit(self, X, y=None)
    _predict(self, X)
    sparse_to_dense(y_sparse, index)  - implemented by sub base classes in skchange

Recommended but optional to implement for a concrete detector:
    dense_to_sparse(y_dense)
    _transform_scores(self, X)
    _update(self, X, y=None)
"""

__author__ = ["Tveten"]
__all__ = ["BaseDetector"]

from sktime.detection.base import BaseDetector as _BaseDetector


class BaseDetector(_BaseDetector):
    """Base class for all detectors in skchange.

    Adjusts the BaseDetector class in sktime to fit the skchange framework as follows:

        * Sets reasonable default values for the tags.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Tveten"],  # author(s) of the object
        "maintainers": ["Tveten"],  # current maintainer(s) of the object
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # str or list of str, package soft dependencies
        # estimator tags
        # --------------
        "object_type": "detector",  # type of object
        "learning_type": "unsupervised",  # supervised, unsupervised
        "task": "None",  # anomaly_detection, change_point_detection, segmentation
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:update": False,
        # todo: distribution_type does not seem to be used - refactor or remove
        "distribution_type": "None",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
    }
