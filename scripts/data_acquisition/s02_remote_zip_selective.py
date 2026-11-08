#!/usr/bin/env python3
"""Stage 02A-0 / P2: selective acquisition of authorised members from a remote ZIP via HTTP Range.

Read-only with respect to the remote server. Safety rules implemented:
  * every request carries a Range header; any response other than 206 with a matching
    Content-Range is ABORTED before the body is read (no accidental full-archive download);
  * TLS certificate validation is left at the Python default (verified);
  * members are selected by exact name from an allow-list; anything ambiguous stops;
  * downloaded member bytes are streamed to disk in chunks, resumable, and every retry is logged;
  * extraction is streamed (zlib), CRC32 is checked against the central directory and the local
    header, and SHA-256 of both the compressed member blob and the extracted CSV are recorded.

Usage:
  python3 -I s02_remote_zip_selective.py list   <url> <out_json>
  python3 -I s02_remote_zip_selective.py fetch  <url> <member_name> <out_dir> <log_file> [<expected_etag>]
  python3 -I s02_remote_zip_selective.py extract <blob_path> <member_json> <out_csv> <log_file>
"""
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

CHUNK = 4 * 1024 * 1024
UA = "IDS-research-stage02a0-audit/1.0 (python-urllib)"


class RangeRefused(Exception):
    pass


def log(path, msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as fh:
            fh.write(line + "\n")


def open_range(url, start, end, logf=None, expected_etag=None):
    """Open a ranged GET [start, end] inclusive. Abort unless 206 + exact Content-Range."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA})
    resp = urllib.request.urlopen(req, timeout=60)
    status = resp.status
    cr = resp.headers.get("Content-Range", "")
    etag = resp.headers.get("ETag")
    clen = resp.headers.get("Content-Length")
    if status != 206:
        resp.close()  # do not read the body
        raise RangeRefused(f"server returned HTTP {status} (Content-Length={clen}) instead of 206; aborted")
    want = f"bytes {start}-{end}/"
    if not cr.startswith(want):
        resp.close()
        raise RangeRefused(f"Content-Range mismatch: got {cr!r}, wanted prefix {want!r}; aborted")
    if clen is not None and int(clen) != end - start + 1:
        resp.close()
        raise RangeRefused(f"Content-Length {clen} != requested {end - start + 1}; aborted")
    if expected_etag and etag != expected_etag:
        resp.close()
        raise RangeRefused(f"ETag changed: {etag} != {expected_etag}; aborted (archive version changed)")
    if logf:
        log(logf, f"RANGE OK 206 {cr} etag={etag}")
    return resp, cr, etag


def get_range(url, start, end, logf=None, expected_etag=None):
    resp, cr, etag = open_range(url, start, end, logf, expected_etag)
    data = resp.read()
    resp.close()
    if len(data) != end - start + 1:
        raise RangeRefused(f"short read {len(data)} for {start}-{end}")
    return data, cr, etag


def total_size(cr):
    return int(cr.rsplit("/", 1)[1])


def parse_central_directory(url, logf=None):
    # 1) tail: EOCD (+ ZIP64 locator)
    probe, cr, etag = get_range(url, 0, 0, logf)  # learn total size from Content-Range
    size = total_size(cr)
    tail_len = min(size, 65536 + 22 + 20)
    tail, _, _ = get_range(url, size - tail_len, size - 1, logf, etag)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ValueError("EOCD not found")
    (_, disk, cd_disk, n_disk, n_total, cd_size, cd_off, clen) = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    zip64 = False
    if cd_off == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x07", 0, i)
        if j < 0:
            raise ValueError("ZIP64 locator not found")
        _, _, eocd64_off, _ = struct.unpack("<IIQI", tail[j:j + 20])
        rec, _, _ = get_range(url, eocd64_off, eocd64_off + 55, logf, etag)
        if rec[:4] != b"PK\x06\x06":
            raise ValueError("ZIP64 EOCD signature mismatch")
        (_, _, _, _, _, _, n_disk, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec[:56])
        zip64 = True
    cd, _, _ = get_range(url, cd_off, cd_off + cd_size - 1, logf, etag)
    members = []
    p = 0
    while p < len(cd):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central header at {p}")
        (sig, vmade, vneed, flag, method, mtime, mdate, crc, csize, usize, nlen, xlen, klen,
         dstart, iattr, eattr, loff) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8" if flag & 0x800 else "cp437")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        # ZIP64 extended info
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            body = extra[q + 4:q + 4 + hlen]
            if hid == 0x0001:
                k = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                if loff == 0xFFFFFFFF:
                    loff = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
            q += 4 + hlen
        y = ((mdate >> 9) & 0x7F) + 1980; mo = (mdate >> 5) & 0xF; d = mdate & 0x1F
        hh = (mtime >> 11) & 0x1F; mi = (mtime >> 5) & 0x3F; ss = (mtime & 0x1F) * 2
        members.append(dict(name=name, flag=flag, method=method, crc32=f"{crc:08x}",
                            compressed_size=csize, uncompressed_size=usize, local_header_offset=loff,
                            dos_mtime=f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mi:02d}:{ss:02d}",
                            encrypted=bool(flag & 1), data_descriptor=bool(flag & 8)))
        p += 46 + nlen + xlen + klen
    return dict(url=url, total_size=size, etag=etag, zip64=zip64, n_entries=n_total,
                cd_offset=cd_off, cd_size=cd_size, members=members)


def fetch_member(url, member, out_dir, logf, expected_etag):
    off = member["local_header_offset"]
    hdr, _, etag = get_range(url, off, off + 29, logf, expected_etag)
    if hdr[:4] != b"PK\x03\x04":
        raise ValueError("local header signature mismatch")
    (_, vneed, flag, method, mt, md, crc, cs, us, nlen, xlen) = struct.unpack("<IHHHHHIIIHH", hdr)
    nm, _, _ = get_range(url, off + 30, off + 30 + nlen - 1, logf, expected_etag)
    if nm.decode("utf-8", "replace") != member["name"]:
        raise ValueError(f"local name {nm!r} != central name {member['name']!r}")
    if method != member["method"]:
        raise ValueError("method mismatch local vs central")
    data_start = off + 30 + nlen + xlen
    data_end = data_start + member["compressed_size"] - 1
    blob = os.path.join(out_dir, member["name"] + ".deflate")
    have = os.path.getsize(blob) if os.path.exists(blob) else 0
    total = member["compressed_size"]
    log(logf, f"FETCH {member['name']} data bytes {data_start}-{data_end} ({total} B); resume from {have}")
    attempt = 0
    while have < total:
        attempt += 1
        try:
            resp, cr, _ = open_range(url, data_start + have, data_end, logf, expected_etag)
            with open(blob, "ab") as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
            resp.close()
        except RangeRefused:
            raise
        except Exception as exc:  # network error -> logged retry/resume
            have = os.path.getsize(blob)
            log(logf, f"RETRY attempt={attempt} error={exc!r}; resuming at {have}")
            if attempt >= 20:
                raise
            time.sleep(min(60, 5 * attempt))
    if have != total:
        raise ValueError(f"blob size {have} != {total}")
    h = hashlib.sha256()
    with open(blob, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    meta = dict(member, local_header=dict(flag=flag, method=method, crc32=f"{crc:08x}", nlen=nlen, xlen=xlen),
                data_start=data_start, data_end=data_end, blob_path=blob, blob_sha256=h.hexdigest(),
                etag=expected_etag, fetched_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), attempts=attempt)
    with open(blob + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    log(logf, f"FETCH DONE {member['name']} blob_sha256={meta['blob_sha256']}")
    return meta


def extract(blob, member, out_csv, logf):
    if member["method"] != 8:
        raise ValueError(f"unsupported method {member['method']}")
    if os.path.exists(out_csv):
        raise FileExistsError(out_csv)
    d = zlib.decompressobj(-15)
    crc = 0
    n = 0
    h = hashlib.sha256()
    with open(blob, "rb") as fi, open(out_csv + ".part", "wb") as fo:
        for chunk in iter(lambda: fi.read(CHUNK), b""):
            out = d.decompress(chunk, CHUNK * 8)
            while True:
                if out:
                    fo.write(out); crc = zlib.crc32(out, crc); h.update(out); n += len(out)
                if not d.unconsumed_tail:
                    break
                out = d.decompress(d.unconsumed_tail, CHUNK * 8)
        out = d.flush()
        if out:
            fo.write(out); crc = zlib.crc32(out, crc); h.update(out); n += len(out)
    ok_crc = f"{crc & 0xffffffff:08x}" == member["crc32"]
    ok_len = n == member["uncompressed_size"]
    ok_eof = d.eof and not d.unused_data
    log(logf, f"EXTRACT {member['name']} bytes={n} crc32={crc & 0xffffffff:08x} expected={member['crc32']} "
              f"crc_ok={ok_crc} size_ok={ok_len} stream_eof={ok_eof} sha256={h.hexdigest()}")
    if not (ok_crc and ok_len and ok_eof):
        os.rename(out_csv + ".part", out_csv + ".FAILED")
        raise ValueError("integrity check failed")
    os.rename(out_csv + ".part", out_csv)
    return dict(bytes=n, crc32=f"{crc & 0xffffffff:08x}", sha256=h.hexdigest())


def main(argv):
    cmd = argv[1]
    if cmd == "list":
        info = parse_central_directory(argv[2])
        with open(argv[3], "w") as fh:
            json.dump(info, fh, indent=1)
        for m in info["members"]:
            print(m["name"], m["method"], m["compressed_size"], m["uncompressed_size"], m["crc32"], m["local_header_offset"])
    elif cmd == "fetch":
        url, name, out_dir, logf = argv[2:6]
        etag = argv[6] if len(argv) > 6 else None
        info = parse_central_directory(url, logf)
        hits = [m for m in info["members"] if m["name"] == name]
        if len(hits) != 1:
            raise SystemExit(f"AMBIGUOUS/MISSING member {name!r}: {len(hits)} matches; stopping")
        if etag and info["etag"] != etag:
            raise SystemExit(f"ETag changed {info['etag']} != {etag}; stopping")
        fetch_member(url, hits[0], out_dir, logf, info["etag"])
    elif cmd == "extract":
        blob, mjson, out_csv, logf = argv[2:6]
        member = json.load(open(mjson))
        res = extract(blob, member, out_csv, logf)
        member["extracted"] = dict(res, path=out_csv)
        with open(mjson, "w") as fh:
            json.dump(member, fh, indent=1)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
