#!/usr/bin/env python3
"""R5-FINAL-v1: Lean NumPy R5 + ZV-1. Implementation of R5_FINAL_SPECIFICATION_v1.md (freeze entry F7).

Approved by the advisor's C4 decision ("APPROVE LEAN NUMPY R5 + ZV-1 FOR FREEZING"). Numerically this is
C4 Candidate A unchanged (correction_package_C4/r5_candidates.py, candidate "A_numpy", zv=True). This module
adds only:
  * the kappa conditioning diagnostic, with a STOP for numerical review (spec section 4);
  * environment and dependency pins, which fail closed;
  * fail-closed value checks on transform (C1 contract, C-F32). These never change a value.

Per numeric training column (69 columns of the frozen 72-feature schema), on the AUTHORISED TRAINING
PARTITION ONLY:
  1. R3 cap (unchanged; v10 rule, C2 lean implementation), then z = sign(x) * log1p(|x|)   (signed log1p kept)
  2. ZV-1: if min(z) == max(z)  (exact equality, no tolerance) -> mu := z[0], sd := 1          [constant]
  3. otherwise mu := numpy.mean(z), sd := numpy.std(z) (ddof = 0) on the WHOLE column as one contiguous 1-D
     float64 array (no chunking); kappa := |mu| / sd (kappa := +inf if sd == 0, which is underflow only)
  4. if kappa > 1e11 for any non-constant column -> STOP: NumericalReviewRequired, evidence preserved,
     NO artefact returned. ZV-1 columns are excluded from kappa.
Held-out (validation/test) data are only ever TRANSFORMED with the saved statistics, never refitted.

Nothing here drops a feature, changes a rule, or switches to another estimator. In particular, Candidate B
(E-R5, math.fsum) is NOT imported. It remains a sensitivity reference only, in correction_package_C4.
"""
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)

SPEC_ID = "R5-FINAL-v1"
VERSION = "r5-final-1"
REQUIRED_NUMPY = "2.4.6"
KAPPA_INVESTIGATION_THRESHOLD = 1e11  # PROVISIONAL investigation threshold; NOT a certified stability bound

# Pinned dependency code (unchanged earlier files). Any byte change -> import fails (fail closed).
DEPENDENCIES = {
    "preprocessing_review/prep_reference.py": "87e25ce15b5053fc9287c47cffa6a788add627fedd1da9834df61f051c62961b",
    "correction_package_C1/prep/prep_v0_2.py": "7ec30d8b83d74bf2ad45cc7e161f9baa9b87546c9cb74cd162b9714f16ec0b5e",
    "correction_package_C2/lean/prep_exact.py": "0144643e601424a0aae9f490d3950544b0522c1adf29323ae16be34a65e6a1f3",
}
# Pinned inputs (unchanged earlier files).
PINNED_INPUTS = {
    "FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv": "6bfc24b8332111fa03729d3597ea4ef8685e11aab999af2906738fd07eee1e9f",
    "preprocessing_review/EXPECTED_CSV_HEADER_91.txt": "c6561a86f03ffbab6b7ef0115d228d30568acdac02f42408ff59a5d89bfc982d",
    "correction_package_C1/provenance/P1_SOURCE_ALLOWLIST.json": "641d1c9d0a130e7bd25e12628178af243497775336c1f92bc03524806ef0df95",
    "next_workstream/CONFIG_GRID_F3_v1.csv": "34b282f13e52a7a5507a450af40513e93da60b27d782245de0f7332c09e7c179",
}
SCHEMA_ORDERED_SHA256 = "e1da1c2ddd702e2bcbd23c54ad1b2b0852bfd26fa9fa154b0a2d8b8a087b1a3f"
SCHEMA_PATH = os.path.join(BRIEF, "FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv")
HEADER_PATH = os.path.join(BRIEF, "preprocessing_review", "EXPECTED_CSV_HEADER_91.txt")
P1_ALLOWLIST_PATH = os.path.join(BRIEF, "correction_package_C1", "provenance", "P1_SOURCE_ALLOWLIST.json")


class DependencyError(RuntimeError):
    pass


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_pins():
    bad = [rel for rel, h in {**DEPENDENCIES, **PINNED_INPUTS}.items() if sha256_path(os.path.join(BRIEF, rel)) != h]
    if bad:
        raise DependencyError(f"pinned file changed: {bad}")
    return True


verify_pins()
for _sub in ("preprocessing_review", os.path.join("correction_package_C1", "prep"), os.path.join("correction_package_C2", "lean")):
    sys.path.insert(0, os.path.join(BRIEF, _sub))
import prep_reference as REF  # noqa: E402  v10 reference (R3/R4 rules, signed log1p, artefact format)
import prep_v0_2 as C1  # noqa: E402       value contract + hash provenance
import prep_exact as C2  # noqa: E402      lean R3/R4 fit and lean transform

for _mod, _rel in ((REF, "preprocessing_review/prep_reference.py"), (C1, "correction_package_C1/prep/prep_v0_2.py"),
                   (C2, "correction_package_C2/lean/prep_exact.py")):
    if os.path.realpath(_mod.__file__) != os.path.realpath(os.path.join(BRIEF, _rel)):
        raise DependencyError(f"module {_mod.__name__} resolved to {_mod.__file__}, expected {_rel}")

PreprocessingError = REF.PreprocessingError


class NumericalReviewRequired(PreprocessingError):
    """kappa above the provisional threshold: the fit is stopped. A separate decision is needed to resume."""

    def __init__(self, message, record, evidence_files):
        super().__init__(message)
        self.record = record
        self.evidence_files = evidence_files


# ----------------------------------------------------------------------------- environment
def _numpy_version():
    return np.__version__


def environment():
    return dict(numpy=_numpy_version(), python=platform.python_version(), machine=platform.machine(),
                platform=platform.platform())


def require_environment():
    if _numpy_version() != REQUIRED_NUMPY:
        raise PreprocessingError(f"NumPy {_numpy_version()} found; R5-FINAL-v1 requires NumPy {REQUIRED_NUMPY} exactly")


def load_contract_verified(schema_path=SCHEMA_PATH):
    if sha256_path(schema_path) != PINNED_INPUTS["FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv"]:
        raise PreprocessingError("feature schema file differs from the pinned 72-feature schema")
    con = C1.load_contract(schema_path)
    if REF.schema_sha256(con["schema"]) != SCHEMA_ORDERED_SHA256:
        raise PreprocessingError("feature order differs from the pinned 72-feature schema order")
    return con


# ----------------------------------------------------------------------------- R5 statistic (one column)
def r5_column_statistics(z):
    """Returns (mu, sd, is_constant, kappa). kappa is None for ZV-1 constant columns."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise PreprocessingError("R5: empty or non-1-D column")
    if not np.isfinite(z).all():
        raise PreprocessingError("R5: non-finite transformed value")
    if z.min() == z.max():                       # ZV-1: exact constancy only
        return float(z[0]), 1.0, True, None
    mu = float(np.mean(z))
    sd_raw = float(np.std(z))                    # population std, ddof = 0
    kappa = math.inf if sd_raw == 0.0 else abs(mu) / sd_raw
    return mu, (1.0 if sd_raw == 0.0 else sd_raw), False, kappa


def kappa_exceeds(kappa):
    """True if the provisional threshold is exceeded (strictly greater; NaN or +inf counts as exceeded)."""
    return not (kappa <= KAPPA_INVESTIGATION_THRESHOLD)


# ----------------------------------------------------------------------------- fit
def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_=" else "_" for ch in s)


def _column_table(numeric, mus, sds, consts, kaps):
    return [dict(column=c, zv1_constant=bool(k0), mean_hex=REF._hx(m), std_hex=REF._hx(s),
                 kappa_hex=(None if k is None else float(k).hex()), kappa=(None if k is None else repr(float(k))),
                 exceeds=(False if k is None else kappa_exceeds(k)))
            for c, m, s, k0, k in zip(numeric, mus, sds, consts, kaps)]


def _stop(art, table, flagged, fold_id, evidence_dir, context):
    record = dict(event="R5_KAPPA_NUMERICAL_REVIEW_STOP", spec=SPEC_ID, version=VERSION, utc=_utc(), fold_id=fold_id,
                  partition=art["partition"], branch=art["branch"], n_train=art["n_train"],
                  kappa_threshold=KAPPA_INVESTIGATION_THRESHOLD, threshold_status="provisional investigation threshold",
                  affected_columns=sorted(flagged), columns=table, zv1_constant_columns=list(art["zv1_constant_columns"]),
                  r3=art["r3"], r4=art["r4"], environment=environment(), context=context,
                  instruction=("Fit stopped for numerical review. No artefact was produced. Do not drop features, change "
                               "preprocessing or substitute another estimator. Assess the affected columns with the "
                               "independent high-precision reference (r5_hp_reference.py). A separate decision is "
                               "required before resuming."))
    files = []
    if evidence_dir is not None:
        base = os.path.join(evidence_dir, f"R5_STOP_{_safe(fold_id)}_{art['branch']}")
        evid = {}
        for c in sorted(flagged):
            p = f"{base}__z__{_safe(c)}.npy"
            with open(p, "xb") as fh:                     # never overwrite earlier evidence
                np.save(fh, flagged[c])
            evid[c] = dict(path=os.path.basename(p), sha256=sha256_path(p), n=int(flagged[c].shape[0]))
            files.append(p)
        record["transformed_training_columns"] = evid
        p = base + ".json"
        with open(p, "x") as fh:
            json.dump(record, fh, indent=1, sort_keys=True)
        files.append(p)
    raise NumericalReviewRequired(
        f"kappa > {KAPPA_INVESTIGATION_THRESHOLD:g} in {sorted(flagged)} (fold {fold_id}): fit stopped for numerical review",
        record, files)


def fit_core(X, schema, branch, partition, fold_id, evidence_dir=None, context=None):
    """R3/R4 (unchanged) + R5-FINAL-v1. Callers on real data must use r5_p1_source.fit_p1_final, which verifies
    the source and passes the verified partition. Returns the artefact, or raises NumericalReviewRequired."""
    require_environment()
    if branch not in ("LR", "TREE"):
        raise PreprocessingError("branch must be LR or TREE")
    if not isinstance(fold_id, str) or not fold_id:
        raise PreprocessingError("fold_id is required")
    art = C2.fit_lean(X, schema, "TREE", partition)              # R3 caps, R4 vocabulary, all v10 checks
    art["branch"] = branch
    art["r5_spec"] = SPEC_ID
    art["fold_id"] = fold_id
    art["zv1_constant_columns"] = []
    art["r5_kappa"] = None
    if branch == "LR":
        pos = {c: j for j, c in enumerate(schema)}
        mus, sds, consts, kaps, flagged = [], [], [], [], {}
        for c in art["numeric"]:
            col = np.ascontiguousarray(X[:, pos[c]], dtype=np.float64)
            if c in art["r3"]:
                col = C2._cap(col, art["r3"][c])
            z = REF.signed_log1p(col)
            mu, sd, const, k = r5_column_statistics(z)
            if const:
                art["zv1_constant_columns"].append(c)
            elif kappa_exceeds(k):
                flagged[c] = z
            mus.append(mu); sds.append(sd); consts.append(const); kaps.append(k)
        art["r5"] = dict(mean=[REF._hx(v) for v in mus], std=[REF._hx(v) for v in sds])
        art["r5_kappa"] = dict(threshold_hex=float(KAPPA_INVESTIGATION_THRESHOLD).hex(),
                               values_hex=[None if k is None else float(k).hex() for k in kaps])
        if flagged:
            _stop(art, _column_table(art["numeric"], mus, sds, consts, kaps), flagged, fold_id, evidence_dir, context)
    art["version"] = VERSION
    art["output_columns"] = REF._output_columns(art)
    return art


def kappa_record(art):
    """Per-fold kappa log for every non-constant numeric training column (ZV-1 columns excluded)."""
    if art.get("branch") != "LR":
        raise PreprocessingError("kappa is defined for the LR branch only (TREE has no R5)")
    mus = [REF._unhx(v) for v in art["r5"]["mean"]]
    sds = [REF._unhx(v) for v in art["r5"]["std"]]
    kaps = [None if v is None else float.fromhex(v) for v in art["r5_kappa"]["values_hex"]]
    consts = [c in art["zv1_constant_columns"] for c in art["numeric"]]
    table = _column_table(art["numeric"], mus, sds, consts, kaps)
    nonconst = [float.fromhex(t["kappa_hex"]) for t in table if not t["zv1_constant"]]
    return dict(event="R5_KAPPA_LOG", spec=SPEC_ID, fold_id=art["fold_id"], partition=art["partition"],
                n_train=art["n_train"], artifact_sha256=REF.artifact_sha256(art), kappa_threshold=KAPPA_INVESTIGATION_THRESHOLD,
                n_nonconstant=len(nonconst), max_kappa=(max(nonconst) if nonconst else None),
                any_exceeds=any(t["exceeds"] for t in table), columns=table, environment=environment(), utc=_utc())


# ----------------------------------------------------------------------------- transform (saved statistics only)
def transform(X, art, contract, out_path=None):
    """Apply the saved artefact to any split. Never refits. Values are identical to C4 Candidate A / C2 lean."""
    require_environment()
    if art.get("r5_spec") != SPEC_ID:
        raise PreprocessingError("artefact was not produced by R5-FINAL-v1")
    if REF.schema_sha256(art["schema"]) != SCHEMA_ORDERED_SHA256 or list(art["schema"]) != list(contract["schema"]):
        raise PreprocessingError("artefact or contract feature order differs from the pinned schema order")
    C1.check_contract(X, contract)
    out, cols = C2.transform_lean(X, art, out_path)
    for j in range(out.shape[1]):                      # C-F32, column at a time (no full-size temporary)
        if np.any(np.abs(out[:, j]) > C1.F32_MAX):
            raise PreprocessingError(f"C-F32 value not representable in float32 in output column {cols[j]!r}")
    return out, cols


def provenance_dict(prov):
    return dataclasses.asdict(prov)
