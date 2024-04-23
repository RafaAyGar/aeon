import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification.convolution_based import RocketClassifier
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)
from mord import LogisticAT
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class OrdinalRocketClassifier(RocketClassifier):
    def _fit(self, X, y):
        y = y.astype(int)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        rocket_transform = self.rocket_transform.lower()
        if rocket_transform == "rocket":
            self._transformer = Rocket(
                num_kernels=self.num_kernels,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif rocket_transform == "minirocket":
            if self.n_dims_ > 1:
                self._transformer = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            else:
                self._transformer = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
        elif rocket_transform == "multirocket":
            if self.n_dims_ > 1:
                self._transformer = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            else:
                self._transformer = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        hyperparameter_space = {"alpha": list(np.logspace(-3, 3, 7))}

        self._scaler = StandardScaler(with_mean=False)
        self._estimator = _clone_estimator(
            (
                GridSearchCV(
                    LogisticAT(),
                    hyperparameter_space,
                    scoring=scorer,
                    error_score="raise",
                    n_jobs=1,
                    cv=StratifiedKFold(n_splits=5),
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )
        self.pipeline_ = make_pipeline(
            self._transformer,
            self._scaler,
            self._estimator,
        )
        self.pipeline_.fit(X, y)
        return self
