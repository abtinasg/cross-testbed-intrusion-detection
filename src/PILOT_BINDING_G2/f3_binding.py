#!/usr/bin/env python3
"""G2: bind the pilot execution plan to the SHA-256-pinned frozen F3 grid (read-only; F3 is never modified).

load_bound_grid() verifies the F3 file hash, parses every row and validates it:
  * exactly the 12 approved config_ids, each once, in complexity_rank order 1..12;
  * family in {LR, HGB, RF}; estimator path fixed by family; preprocessing variant fixed by family
    (LR -> P-LR -> branch LR; HGB/RF -> P-TREE -> branch TREE);
  * hyperparameters_json is a JSON object with no duplicate keys;
  * each configuration's canonical SHA-256 equals the value pinned below (independent of the file hash).
expected_plan(phase, selected) gives the exact (config_id, fold_id) key set for an approved CV phase, built with
guard v2.1 expected_keys from the F3 config_ids only (never from caller-supplied ids), and checks its size.
"""
import csv
import dataclasses
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BRIEF, "correction_package_C1", "guard"))
import guard_v2_1 as G  # noqa: E402

F3_PATH = os.path.join(BRIEF, "next_workstream", "CONFIG_GRID_F3_v1.csv")
F3_SHA256 = "34b282f13e52a7a5507a450af40513e93da60b27d782245de0f7332c09e7c179"
APPROVED_IDS = ("LR-01", "LR-02", "LR-03", "HGB-01", "HGB-05", "HGB-02", "HGB-03", "HGB-04", "RF-01", "RF-02", "RF-03", "RF-04")
FAMILY_ESTIMATOR = {"LR": "sklearn.linear_model.LogisticRegression",
                    "HGB": "sklearn.ensemble.HistGradientBoostingClassifier",
                    "RF": "sklearn.ensemble.RandomForestClassifier"}
FAMILY_VARIANT = {"LR": "P-LR", "HGB": "P-TREE", "RF": "P-TREE"}
VARIANT_BRANCH = {"P-LR": "LR", "P-TREE": "TREE"}
PHASE_SIZES = {"flat": 60, "tb": 20, "nested": 240}   # 12 x 5; 1 x 20 (selected); 12 x 20
# Canonical per-configuration SHA-256 (config_sha256 below), computed from the frozen F3 v1 file at G2 creation.
EXPECTED_CONFIG_SHA256 = {}   # filled by the pinned table at the end of this module


class BindingError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class ConfigSpec:
    complexity_rank: int
    config_id: str
    family: str
    estimator: str
    hyperparameters_json: str        # exact frozen text
    preprocessing_variant: str

    @property
    def hyperparameters(self):
        return json.loads(self.hyperparameters_json)

    @property
    def branch(self):
        return VARIANT_BRANCH[self.preprocessing_variant]

    @property
    def config_sha256(self):
        canon = json.dumps(dict(config_id=self.config_id, family=self.family, estimator=self.estimator,
                                hyperparameters=self.hyperparameters, preprocessing_variant=self.preprocessing_variant,
                                complexity_rank=self.complexity_rank), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()


def _no_dup_object(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise BindingError(f"duplicate hyperparameter keys {sorted(k for k in keys if keys.count(k) > 1)}")
    return dict(pairs)


def validate_rows(rows, check_pinned=True):
    """Structural validation of parsed F3 rows (list of dicts with the six F3 columns)."""
    cols = {"complexity_rank", "config_id", "family", "estimator", "hyperparameters_json", "preprocessing_variant"}
    specs = []
    for r in rows:
        if set(r) != cols:
            raise BindingError(f"unexpected F3 columns {sorted(set(r) ^ cols)}")
        hp = json.loads(r["hyperparameters_json"], object_pairs_hook=_no_dup_object)
        if not isinstance(hp, dict) or not hp:
            raise BindingError(f"{r['config_id']}: hyperparameters must be a non-empty JSON object")
        fam = r["family"]
        if fam not in FAMILY_ESTIMATOR:
            raise BindingError(f"{r['config_id']}: unknown family {fam!r}")
        if r["estimator"] != FAMILY_ESTIMATOR[fam]:
            raise BindingError(f"{r['config_id']}: estimator {r['estimator']!r} does not match family {fam}")
        if r["preprocessing_variant"] != FAMILY_VARIANT[fam]:
            raise BindingError(f"{r['config_id']}: preprocessing variant {r['preprocessing_variant']!r} does not match family {fam}")
        if not r["config_id"].startswith(fam + "-"):
            raise BindingError(f"{r['config_id']}: id prefix does not match family {fam}")
        specs.append(ConfigSpec(int(r["complexity_rank"]), r["config_id"], fam, r["estimator"], r["hyperparameters_json"],
                                r["preprocessing_variant"]))
    ids = [s.config_id for s in specs]
    if len(ids) != len(set(ids)):
        raise BindingError(f"duplicate config_id {sorted(i for i in ids if ids.count(i) > 1)}")
    missing, unexpected = sorted(set(APPROVED_IDS) - set(ids)), sorted(set(ids) - set(APPROVED_IDS))
    if missing or unexpected or len(ids) != 12:
        raise BindingError(f"F3 must contain exactly the 12 approved configurations; missing {missing}, unexpected {unexpected}")
    if tuple(ids) != APPROVED_IDS or [s.complexity_rank for s in specs] != list(range(1, 13)):
        raise BindingError("F3 order / complexity_rank differs from the frozen order")
    if check_pinned:
        bad = [s.config_id for s in specs if s.config_sha256 != EXPECTED_CONFIG_SHA256[s.config_id]]
        if bad:
            raise BindingError(f"configuration content differs from the pinned F3 binding: {bad}")
    return {s.config_id: s for s in specs}


def load_bound_grid(path=F3_PATH):
    with open(path, "rb") as fh:
        raw = fh.read()
    if hashlib.sha256(raw).hexdigest() != F3_SHA256:
        raise BindingError("F3 grid file differs from the frozen F3 v1 (SHA-256 mismatch)")
    rows = list(csv.DictReader(l for l in raw.decode("utf-8").splitlines() if not l.startswith("#")))
    grid = validate_rows(rows)
    if G.grid_config_ids(path) != list(APPROVED_IDS):            # guard v2.1's own reading agrees
        raise BindingError("guard v2.1 reads different config_ids from F3")
    return grid


def expected_plan(grid, phase, selected=None):
    if phase not in PHASE_SIZES:
        raise BindingError(f"unknown CV phase {phase!r}")
    if phase == "tb" and selected not in grid:
        raise BindingError(f"T-b phase requires a selected config_id from F3, got {selected!r}")
    if phase != "tb" and selected is not None:
        raise BindingError("selected is only used in the T-b phase")
    keys = G.expected_keys(phase, list(grid), selected)
    if len(keys) != PHASE_SIZES[phase]:
        raise BindingError(f"{phase}: {len(keys)} keys, expected {PHASE_SIZES[phase]}")
    return keys


EXPECTED_CONFIG_SHA256.update({
    # generated once from frozen F3 v1 at G2 creation (f3_binding_pins.txt)
    "LR-01": "cbb6a0df6b71f048f703871910be352b51cee587055a5792836b052888ac9373",
    "LR-02": "ad471d6f46c3e59845d82a61b5783942c249e385ad4ba3b329093887b676305e",
    "LR-03": "6233c7292d0d1260600115f03536aed573723be27f76a61d30b76a46fc2e7c3e",
    "HGB-01": "bd85ce22cc26f3d387f936e01cfa7065301e538875b7c9c784e9255a85521e3a",
    "HGB-05": "ffa1d05b697100f4330ab49480997862e0fab557c74d1e9e0f2a0fccac6db01d",
    "HGB-02": "e6bbc140d53f19d637b86ff510f6144a233552a9f3b8b9dd4f753a1d49b78af2",
    "HGB-03": "07ac1c5edc923123f80985e344d55c1e2bbc3f75709dfc7bd7f7dde1ce79bbfc",
    "HGB-04": "62f38e15b41e452d49433557b667ccd52b36c963bd66640257014c05c5c1426c",
    "RF-01": "dc7f81a10346b53067d88da510f62605bca0bb17178baab8b7c7edab739d2c1d",
    "RF-02": "0d8fe83ce23ec89578ae5c637c11caf8dbd6044b8dbc6824f007767c7032a045",
    "RF-03": "8a93f95e96226ac2179c6ac658328d26d8caaca14807bdce6630e8fb89be2c2f",
    "RF-04": "f8623337c53c0cdd951b2c737a570a3eac0f93dcd65c1cc3e6876268d01c1ced",
})
