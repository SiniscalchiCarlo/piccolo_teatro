"""Utilities for training an XGBoost model.

The previous version of this module contained a minimal wrapper around
``XGBRegressor``.  In order to obtain more reliable predictions we now
incorporate a cross‑validated hyper‑parameter search together with
early stopping on a validation set.  The function exposed by this
module returns both the fitted model and the information gathered
during the training phase so that callers can log or further analyse
the results.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from piccolo_teatro.config import TrainConfig


# Instantiate the training configuration once so that the same values
# are reused across function calls.
train_conf = TrainConfig()


def train_xgb(
    train_X,
    train_Y,
    validation_X=None,
    validation_Y=None,
) -> Tuple[XGBRegressor, Dict, Optional[float]]:
    """Train an ``XGBRegressor`` using robust practices.

    Parameters
    ----------
    train_X, train_Y:
        Feature matrix and target vector used for fitting the model.
    validation_X, validation_Y:
        Optional hold‑out set employed both for early stopping and for
        reporting the final validation error.  When not provided the
        model is trained on the full dataset without early stopping.

    Returns
    -------
    model : ``XGBRegressor``
        The fitted model using the best hyper‑parameters discovered.
    best_params : ``dict``
        Hyper‑parameter configuration chosen by the search procedure.
    val_mae : ``float`` or ``None``
        Mean absolute error on the validation set.  ``None`` when no
        validation data is supplied.

    Notes
    -----
    The function first performs a hyper‑parameter search using a
    time‑series aware cross validation.  The best configuration is then
    used to train a final model with optional early stopping.  This
    approach is considerably more robust than fitting a model with a
    single, fixed set of parameters.
    """

    # ------------------------------------------------------------------
    # 1) Hyper‑parameter optimisation
    # ------------------------------------------------------------------
    # Base estimator with objective taken from the configuration.
    base_model = XGBRegressor(
        objective=train_conf.parameters["objective"],
        random_state=42,
        tree_method="hist",  # faster training on CPUs
    )

    # TimeSeriesSplit preserves the temporal order of observations, an
    # essential requirement when dealing with sequential data.
    tscv = TimeSeriesSplit(n_splits=train_conf.cv_splits)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=train_conf.param_grid,
        n_iter=20,  # number of sampled combinations
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,  # use all available cores
        verbose=1,
    )

    search.fit(train_X, train_Y)

    # Merge default parameters with the ones found by the search so
    # that unspecified settings (such as the objective) are preserved.
    best_params = {**train_conf.parameters, **search.best_params_}

    # ------------------------------------------------------------------
    # 2) Train final model with early stopping
    # ------------------------------------------------------------------
    model = XGBRegressor(**best_params, random_state=42)

    val_mae: Optional[float] = None
    if validation_X is not None and validation_Y is not None:
        # Early stopping helps preventing over‑fitting on sequential
        # data by monitoring the error on a validation set.
        model.fit(
            train_X,
            train_Y,
            eval_set=[(validation_X, validation_Y)],
            early_stopping_rounds=20,
            verbose=False,
        )

        # Calculate the mean absolute error on the validation set to
        # provide a quantitative estimate of the model performance.
        val_predictions = model.predict(validation_X)
        val_mae = mean_absolute_error(validation_Y, val_predictions)
    else:
        # When no validation set is supplied we simply fit on the full
        # data without early stopping.
        model.fit(train_X, train_Y)

    return model, best_params, val_mae


__all__ = ["train_xgb"]

