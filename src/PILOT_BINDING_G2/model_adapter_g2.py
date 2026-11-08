#!/usr/bin/env python3
"""G2 PROPOSED real-model adapter (NOT frozen; training NOT authorised).

Maps each frozen F3 configuration to its scikit-learn 1.9.0 estimator:
  LR  -> sklearn.linear_model.LogisticRegression           (branch LR,   P-LR)
  HGB -> sklearn.ensemble.HistGradientBoostingClassifier    (branch TREE, P-TREE)
  RF  -> sklearn.ensemble.RandomForestClassifier            (branch TREE, P-TREE)
build_estimator(cfg) constructs the estimator WITHOUT fitting, and checks:
  * exact environment pins (sklearn, numpy, scipy, joblib, threadpoolctl, python);
  * scikit-learn's own constructor validation (_validate_params);
  * get_params(deep=False) equals the full pinned parameter set in ADAPTER_PINNED_PARAMS_v1_PROPOSED.json
    (this also pins every library default not written in F3), and the F3 dictionary is a sub-dictionary of it.
train_and_score() is the intended real-model step: fit(X_train, y_train); score = predict_proba(X_val)[:, 1]
(float64, unrounded). It refuses unconditionally while PILOT_TRAINING_AUTHORIZED is False. No data is touched.
"""
import hashlib
import importlib
import json
import os
import platform

import f3_binding as B

HERE = os.path.dirname(os.path.abspath(__file__))
PINS_PATH = os.path.join(HERE, "ADAPTER_PINNED_PARAMS_v1_PROPOSED.json")
PINS_SHA256 = "42b47d9c2c732949528a578841c69bc2f65006fe16a550e510a2b9b8e4d5b79a"
PILOT_TRAINING_AUTHORIZED = False
ESTIMATOR_CLASSES = dict(B.FAMILY_ESTIMATOR)


class AdapterError(Exception):
    pass


def load_pins():
    with open(PINS_PATH, "rb") as fh:
        raw = fh.read()
    if hashlib.sha256(raw).hexdigest() != PINS_SHA256:
        raise AdapterError("adapter parameter pins changed")
    return json.loads(raw)


def check_environment(pins=None):
    pins = pins or load_pins()
    import joblib, numpy, scipy, sklearn, threadpoolctl
    found = dict(python=platform.python_version(), numpy=numpy.__version__, scikit_learn=sklearn.__version__,
                 scipy=scipy.__version__, joblib=joblib.__version__, threadpoolctl=threadpoolctl.__version__)
    if found != pins["environment"]:
        raise AdapterError(f"environment {found} differs from pinned {pins['environment']}")
    return found


def estimator_class(cfg):
    if not isinstance(cfg, B.ConfigSpec):
        raise AdapterError("a ConfigSpec from f3_binding.load_bound_grid is required")
    path = ESTIMATOR_CLASSES.get(cfg.family)
    if path is None or cfg.estimator != path:
        raise AdapterError(f"{cfg.config_id}: estimator {cfg.estimator!r} not allowed for family {cfg.family!r}")
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def build_estimator(cfg, pins=None):
    """Construct and validate (no fit)."""
    pins = pins or load_pins()
    check_environment(pins)
    entry = pins["configs"].get(cfg.config_id) if isinstance(cfg, B.ConfigSpec) else None
    if entry is None:
        raise AdapterError(f"no pinned adapter entry for {getattr(cfg, 'config_id', cfg)!r}")
    if entry["config_sha256"] != cfg.config_sha256 or entry["branch"] != cfg.branch or entry["estimator"] != cfg.estimator:
        raise AdapterError(f"{cfg.config_id}: adapter entry does not match the F3 binding")
    est = estimator_class(cfg)(**cfg.hyperparameters)
    est._validate_params()
    params = est.get_params(deep=False)
    if params != entry["full_constructor_params"]:
        diff = {k: (params.get(k), entry["full_constructor_params"].get(k))
                for k in set(params) | set(entry["full_constructor_params"]) if params.get(k) != entry["full_constructor_params"].get(k)}
        raise AdapterError(f"{cfg.config_id}: constructed parameters differ from pins: {diff}")
    if any(params[k] != v for k, v in cfg.hyperparameters.items()):
        raise AdapterError(f"{cfg.config_id}: F3 hyperparameter not applied")
    return est


def train_and_score(cfg, X_train, y_train, X_val):
    if not PILOT_TRAINING_AUTHORIZED:
        raise AdapterError("NOT_AUTHORIZED: real-model training is not authorised (Stage 02A-1 not started)")
    est = build_estimator(cfg)                                    # pragma: no cover  (unreachable until authorised)
    est.fit(X_train, y_train)
    return est.predict_proba(X_val)[:, 1].astype("float64", copy=False)
