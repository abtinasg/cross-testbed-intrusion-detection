#!/usr/bin/env python3
"""Source-verified construction of P1 training matrices + the authoritative real-data R5 entry point (F7).

Advisor C4 decision, item 4: before any real-data fit, verify that the ACTUAL matrix rows come from the
approved P1 file hashes and the correct training-day partition. Reject any mismatch in source, fold
membership, feature order or training-row count. Do not rely only on a manually supplied provenance object.

How this module does that:
  * load_day_block(path, ...) hashes the file in full (pre-hash) and maps the hash through the P1 allow-list,
    which rejects P2 and unknown hashes such as P3. It then parses the SAME bytes while hashing them again
    (streaming hash). The two hashes and the byte count must agree with the allow-list.
    The header must equal the frozen 91-column header. The 72 features are selected by name in the frozen
    schema order. `id` must equal the file row number (1..N, the Stage 02A-0 finding), which fixes file order.
    The row count must equal the independently pinned Stage 02A-0 count for that day. The C1 value contract
    is applied.
    The parsed matrix is made read-only, hashed, and registered in a module-private registry.
  * assemble_training_matrix(blocks, fold, ...) requires exactly the training days of the fold (CV v2: leave-
    one-day-out, inner folds by day), orders them Mon -> Fri (the defined training-row order) and concatenates.
  * fit_p1_final(tm, ...) re-derives everything from the data (row counts, per-day and total matrix hashes,
    registry membership, feature order, allow-list identity, fold membership), BUILDS the provenance itself
    from the verified blocks (a caller-supplied TrainingProvenance is not accepted), and only then fits R5.
Residual risk: in-process tampering with this module's registry cannot be prevented in Python. It would
be deliberate and is out of scope.
"""
import csv
import dataclasses
import hashlib
import io
import json
import os

import numpy as np

import r5_final as R

REF, C1 = R.REF, R.C1
PreprocessingError = R.PreprocessingError
DAYS = C1.DAYS  # ("Mon", "Tue", "Wed", "Thu", "Fri")
TARGETS = ("primary", "sensitivity")

# Independent anchors from Stage 02A-0 (04_DATA_QUALITY_REPORT.md section 3; results/P1_<day>.summary.json):
# complete-file rows per day, and rows with Attempted Category != -1 (excluded under the sensitivity target).
P1_FILE_ROWS = (("Mon", 371624), ("Tue", 322078), ("Wed", 496641), ("Thu", 362076), ("Fri", 547557))
P1_ATTEMPTED_ROWS = (("Mon", 0), ("Tue", 39), ("Wed", 5876), ("Thu", 1997), ("Fri", 4067))
REAL_P1_FILE_SHA256 = frozenset({
    "51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff",  # Mon
    "e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c",  # Tue
    "bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2",  # Wed
    "78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482",  # Thu
    "ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0",  # Fri
})


@dataclasses.dataclass(frozen=True)
class SourceProfile:
    name: str
    allowlist_sha256: str
    file_rows: tuple          # ((day, rows), ...) complete-file rows
    attempted_rows: tuple     # ((day, rows), ...) rows with Attempted Category != -1

    def expected_file_rows(self, day):
        return dict(self.file_rows)[day]

    def expected_rows(self, day, target):
        n = dict(self.file_rows)[day]
        return n if target == "primary" else n - dict(self.attempted_rows)[day]


P1_PROFILE = SourceProfile("P1", R.PINNED_INPUTS["correction_package_C1/provenance/P1_SOURCE_ALLOWLIST.json"],
                           P1_FILE_ROWS, P1_ATTEMPTED_ROWS)


@dataclasses.dataclass(frozen=True)
class FoldSpec:
    """CV v2 section 2: outer leave-one-day-out; inner leave-one-day-out over the outer training days."""
    outer: str
    inner: str = None

    def __post_init__(self):
        if self.outer not in DAYS:
            raise PreprocessingError(f"unknown outer held-out day {self.outer!r}")
        if self.inner is not None and (self.inner not in DAYS or self.inner == self.outer):
            raise PreprocessingError(f"invalid inner held-out day {self.inner!r} for outer {self.outer!r}")

    def training_days(self):
        return tuple(d for d in DAYS if d not in (self.outer, self.inner))

    @property
    def fold_id(self):
        return f"outer={self.outer}" + ("" if self.inner is None else f"/inner={self.inner}")


@dataclasses.dataclass(frozen=True, eq=False)
class DayBlock:
    day: str
    file_sha256: str
    file_bytes: int
    target: str
    n_file_rows: int
    n_rows: int
    X: np.ndarray
    ids: np.ndarray
    labels: tuple
    attempted: np.ndarray
    x_sha256: str
    ids_sha256: str
    labels_sha256: str
    profile_name: str
    allowlist_sha256: str
    feature_order_sha256: str


@dataclasses.dataclass(frozen=True, eq=False)
class TrainingMatrix:
    fold: FoldSpec
    target: str
    days: tuple
    file_sha256: tuple
    n_rows_by_day: tuple
    block_x_sha256: tuple
    block_ids_sha256: tuple
    block_labels_sha256: tuple
    x_sha256: str
    X: np.ndarray
    ids: np.ndarray
    labels: tuple
    attempted: np.ndarray
    profile_name: str
    allowlist_sha256: str
    feature_order_sha256: str


_REGISTRY = {}  # x_sha256 -> registration tuple, written only by load_day_block / assemble_training_matrix


# ----------------------------------------------------------------------------- hashing helpers
class _HashingRaw(io.RawIOBase):
    def __init__(self, fh):
        self._fh, self.h, self.nbytes = fh, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        n = self._fh.readinto(b)
        if n:
            self.h.update(memoryview(b)[:n])
            self.nbytes += n
        return n


def _sha_array(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def _sha_labels(labels):
    return hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()


def load_expected_header(path=R.HEADER_PATH):
    if R.sha256_path(path) != R.PINNED_INPUTS["preprocessing_review/EXPECTED_CSV_HEADER_91.txt"]:
        raise PreprocessingError("expected CSV header file differs from the pinned 91-column header")
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def check_profile(profile, allowlist, hashes=()):
    """Real P1 files can only be used under the pinned P1 profile + pinned allow-list, and vice versa."""
    if not isinstance(profile, SourceProfile):
        raise PreprocessingError("a SourceProfile is required")
    if profile.allowlist_sha256 != allowlist["_sha256"]:
        raise PreprocessingError("allow-list does not match the source profile")
    real = profile == P1_PROFILE
    if (allowlist["_sha256"] == P1_PROFILE.allowlist_sha256) != real or (profile.name == "P1") != real:
        raise PreprocessingError("the pinned P1 allow-list can be used only with the pinned P1 profile")
    if not real and any(h in REAL_P1_FILE_SHA256 for h in hashes):
        raise PreprocessingError("a real P1 file hash appears under a non-P1 (test) profile")
    return True


# ----------------------------------------------------------------------------- day blocks
def load_day_block(path, allowlist, contract, target="primary", profile=P1_PROFILE, header_path=R.HEADER_PATH):
    if target not in TARGETS:
        raise PreprocessingError(f"unknown target {target!r}")
    schema = contract["schema"]
    if REF.schema_sha256(schema) != R.SCHEMA_ORDERED_SHA256:
        raise PreprocessingError("contract feature order differs from the pinned 72-feature schema order")
    expected_header = load_expected_header(header_path)
    h0 = C1.sha256_file(path)                                   # pre-hash (full file)
    check_profile(profile, allowlist, (h0,))
    prov = C1.provenance_from_hashes([h0], allowlist)           # rejects forbidden (P2) and unknown (P3) hashes
    day = prov.days[0]
    n_file_exp = profile.expected_file_rows(day)
    n_exp = profile.expected_rows(day, target)
    X = np.empty((n_exp, len(schema)), dtype=np.float64)
    ids = np.empty(n_exp, dtype=np.int64)
    att = np.empty(n_exp, dtype=np.int64)
    labels = []
    with open(path, "rb", buffering=0) as raw_fh:
        raw = _HashingRaw(raw_fh)
        txt = io.TextIOWrapper(io.BufferedReader(raw, 1 << 20), encoding="utf-8", newline="")
        rd = csv.reader(txt)
        header = next(rd, None)
        if header is None:
            raise PreprocessingError("empty file")
        REF.validate_header(header, expected_header)            # names, count and order
        idx = {h: i for i, h in enumerate(header)}
        cols = [idx[c] for c in schema]
        i_id, i_lab, i_att, nh = idx["id"], idx["Label"], idx["Attempted Category"], len(header)
        r = k = 0
        for row in rd:
            if len(row) != nh:
                raise PreprocessingError(f"{day} row {r + 1}: {len(row)} fields, expected {nh}")
            t = row[i_id].strip()
            if not t.isdigit() or int(t) != r + 1:
                raise PreprocessingError(f"{day} row {r + 1}: id {row[i_id]!r} != file row (file order not preserved)")
            a = float(row[i_att])
            if not np.isfinite(a) or a != int(a):
                raise PreprocessingError(f"{day} row {r + 1}: invalid Attempted Category {row[i_att]!r}")
            lab = row[i_lab].strip()
            if not lab:
                raise PreprocessingError(f"{day} row {r + 1}: empty Label")
            r += 1
            if target == "sensitivity" and int(a) != -1:
                continue
            if k >= n_exp:
                raise PreprocessingError(f"{day}: more training rows than the pinned count {n_exp}")
            X[k] = [C1.parse_token_strict(row[j], c, contract) for j, c in zip(cols, schema)]
            ids[k] = r; att[k] = int(a); labels.append(lab)
            k += 1
        txt.read()                                              # drain: every byte goes through the hash
    if raw.h.hexdigest() != h0:
        raise PreprocessingError(f"{day}: file changed between hashing and parsing (streaming hash mismatch)")
    if raw.nbytes != allowlist["P1"][h0].get("bytes", raw.nbytes):
        raise PreprocessingError(f"{day}: byte count {raw.nbytes} != allow-list")
    if r != n_file_exp:
        raise PreprocessingError(f"{day}: {r} file rows, pinned Stage 02A-0 count is {n_file_exp}")
    if k != n_exp:
        raise PreprocessingError(f"{day}: {k} training rows for target {target}, pinned count is {n_exp}")
    C1.check_contract(X, contract)
    for a in (X, ids, att):
        a.setflags(write=False)
    labels = tuple(labels)
    blk = DayBlock(day, h0, raw.nbytes, target, r, k, X, ids, labels, att, _sha_array(X), _sha_array(ids),
                   _sha_labels(labels), profile.name, allowlist["_sha256"], REF.schema_sha256(schema))
    _REGISTRY[blk.x_sha256] = ("day", day, h0, target, k, blk.ids_sha256, blk.labels_sha256, profile.name, allowlist["_sha256"])
    return blk


def _check_block(b, target, profile, allowlist):
    if not isinstance(b, DayBlock):
        raise PreprocessingError("training input must be DayBlock objects from load_day_block")
    reg = _REGISTRY.get(b.x_sha256)
    if reg != ("day", b.day, b.file_sha256, target, b.n_rows, b.ids_sha256, b.labels_sha256, profile.name, allowlist["_sha256"]):
        raise PreprocessingError(f"{b.day}: block not registered by load_day_block (or registration mismatch)")
    if _sha_array(b.X) != b.x_sha256 or _sha_array(b.ids) != b.ids_sha256 or _sha_labels(b.labels) != b.labels_sha256:
        raise PreprocessingError(f"{b.day}: block content differs from what was parsed from the source file")


def assemble_training_matrix(blocks, fold, allowlist, contract, target="primary", profile=P1_PROFILE):
    if not isinstance(fold, FoldSpec):
        raise PreprocessingError("a FoldSpec is required")
    check_profile(profile, allowlist, tuple(getattr(b, "file_sha256", "") for b in blocks))
    for b in blocks:
        _check_block(b, target, profile, allowlist)
    days = [b.day for b in blocks]
    if len(set(days)) != len(days):
        raise PreprocessingError(f"duplicate training day in {days}")
    need = fold.training_days()
    if set(days) != set(need):
        raise PreprocessingError(f"fold {fold.fold_id}: training days {sorted(days, key=DAYS.index)} != required {list(need)}")
    if REF.schema_sha256(contract["schema"]) != R.SCHEMA_ORDERED_SHA256:
        raise PreprocessingError("contract feature order differs from the pinned schema order")
    by = {b.day: b for b in blocks}
    ordered = [by[d] for d in need]                            # defined training-row order: Mon -> Fri, file order
    X = np.concatenate([b.X for b in ordered], axis=0)
    ids = np.concatenate([b.ids for b in ordered]); att = np.concatenate([b.attempted for b in ordered])
    labels = tuple(l for b in ordered for l in b.labels)
    for a in (X, ids, att):
        a.setflags(write=False)
    tm = TrainingMatrix(fold, target, need, tuple(b.file_sha256 for b in ordered), tuple(b.n_rows for b in ordered),
                        tuple(b.x_sha256 for b in ordered), tuple(b.ids_sha256 for b in ordered),
                        tuple(b.labels_sha256 for b in ordered), _sha_array(X), X, ids, labels, att,
                        profile.name, allowlist["_sha256"], REF.schema_sha256(contract["schema"]))
    _REGISTRY[tm.x_sha256] = ("matrix", fold.fold_id, target, tm.file_sha256, tm.block_x_sha256, profile.name, allowlist["_sha256"])
    return tm


def verify_training_matrix(tm, fold, allowlist, contract, profile=P1_PROFILE):
    """Re-derive and check everything from the data. Returns (TrainingProvenance built here, verification record)."""
    if not isinstance(tm, TrainingMatrix):
        raise PreprocessingError("fit_p1_final requires a TrainingMatrix from assemble_training_matrix "
                                 "(raw matrices, string tags and manually supplied provenance are not accepted)")
    if not isinstance(fold, FoldSpec) or tm.fold != fold:
        raise PreprocessingError(f"training matrix was assembled for fold {getattr(tm.fold, 'fold_id', None)}, not {getattr(fold, 'fold_id', fold)}")
    check_profile(profile, allowlist, tm.file_sha256)
    if tm.profile_name != profile.name or tm.allowlist_sha256 != allowlist["_sha256"]:
        raise PreprocessingError("training matrix was built under a different profile or allow-list")
    need = fold.training_days()
    if tuple(tm.days) != need:
        raise PreprocessingError(f"fold membership/order mismatch: {tm.days} != {need}")
    prov = C1.provenance_from_hashes(list(tm.file_sha256), allowlist)   # source hashes -> days, via the allow-list
    if prov.days != need or prov.file_sha256 != tuple(tm.file_sha256):
        raise PreprocessingError("source hashes do not map to the fold's training days")
    if REF.schema_sha256(contract["schema"]) != R.SCHEMA_ORDERED_SHA256 or tm.feature_order_sha256 != R.SCHEMA_ORDERED_SHA256:
        raise PreprocessingError("feature order mismatch")
    if tm.X.ndim != 2 or tm.X.shape[1] != 72:
        raise PreprocessingError("training matrix must have exactly 72 feature columns")
    exp = tuple(profile.expected_rows(d, tm.target) for d in need)
    if tuple(tm.n_rows_by_day) != exp or tm.X.shape[0] != sum(exp) or tm.ids.shape[0] != sum(exp) or len(tm.labels) != sum(exp):
        raise PreprocessingError(f"training-row count mismatch: {tm.n_rows_by_day} / {tm.X.shape[0]} vs pinned {exp}")
    if _REGISTRY.get(tm.x_sha256) != ("matrix", fold.fold_id, tm.target, tuple(tm.file_sha256), tuple(tm.block_x_sha256), profile.name, allowlist["_sha256"]):
        raise PreprocessingError("training matrix not registered by assemble_training_matrix")
    if _sha_array(tm.X) != tm.x_sha256:
        raise PreprocessingError("training matrix content changed after assembly")
    a = 0
    for d, h, n, hx, hi, hl in zip(need, tm.file_sha256, tm.n_rows_by_day, tm.block_x_sha256, tm.block_ids_sha256, tm.block_labels_sha256):
        if _REGISTRY.get(hx) != ("day", d, h, tm.target, n, hi, hl, profile.name, allowlist["_sha256"]):
            raise PreprocessingError(f"{d}: rows not traceable to a registered source block")
        if _sha_array(tm.X[a:a + n]) != hx or _sha_array(tm.ids[a:a + n]) != hi or _sha_labels(tm.labels[a:a + n]) != hl:
            raise PreprocessingError(f"{d}: matrix rows differ from the rows parsed from the source file")
        a += n
    record = dict(fold_id=fold.fold_id, target=tm.target, profile=profile.name, allowlist_sha256=allowlist["_sha256"],
                  days=list(need), file_sha256=list(tm.file_sha256), n_rows_by_day=list(tm.n_rows_by_day),
                  n_train=int(tm.X.shape[0]), x_sha256=tm.x_sha256, block_x_sha256=list(tm.block_x_sha256),
                  feature_order_sha256=tm.feature_order_sha256, row_order="Mon->Fri, file (id) order")
    return prov, record


# ----------------------------------------------------------------------------- authoritative real-data entry point
def fit_p1_final(tm, contract, branch, allowlist, fold, evidence_dir, profile=P1_PROFILE):
    if evidence_dir is None or not os.path.isdir(evidence_dir):
        raise PreprocessingError("an existing evidence directory is required (kappa log / stop evidence)")
    prov, ver = verify_training_matrix(tm, fold, allowlist, contract, profile)
    C1._validate_provenance(prov, allowlist, fold.outer)
    if fold.inner is not None:
        C1._validate_provenance(prov, allowlist, fold.inner)
    C1.check_contract(tm.X, contract)
    partition = "P1:" + ",".join(prov.days)
    ctx = dict(provenance=R.provenance_dict(prov), source_verification=ver)
    art = R.fit_core(tm.X, contract["schema"], branch, partition, fold.fold_id, evidence_dir, ctx)
    art["provenance"] = R.provenance_dict(prov)
    art["source_verification"] = ver
    if branch == "LR":
        p = os.path.join(evidence_dir, f"R5_KAPPA_{R._safe(fold.fold_id)}_{tm.target}.json")
        with open(p, "x") as fh:                               # never overwrite an earlier record
            json.dump(R.kappa_record(art), fh, indent=1, sort_keys=True)
    return art
