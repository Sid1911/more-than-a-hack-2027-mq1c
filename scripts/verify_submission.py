"""Check that a regenerated submission file is byte-for-byte identical to the original.

    python scripts/verify_submission.py ORIGINAL.csv REGENERATED.csv

Reports, for both files: SHA-256, size, encoding, newline convention, header, row count. Then,
if the bytes differ, where they first differ: header, row count, row order (frame column) or
values. Exits 0 only when the two files are identical byte for byte. Needs no third-party
packages, so it runs before anything is installed.
"""

import hashlib
import sys

EXPECTED_HEADER = "frame,label"
EXPECTED_ROWS = 199319
LABELS = {"H", "A", "D", "X"}


def describe(path):
    raw = open(path, "rb").read()
    info = {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        text = raw.decode("ascii")
        info["encoding"] = "ASCII (also valid UTF-8)"
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        info["encoding"] = "NOT ASCII"
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    info["newlines"] = ("LF" if lf and not crlf else "CRLF" if crlf and not lf
                        else "mixed" if crlf and lf else "none")
    info["bom"] = raw.startswith(b"\xef\xbb\xbf")
    info["ends_with_newline"] = raw.endswith(b"\n")
    lines = text.splitlines()
    info["header"] = lines[0] if lines else ""
    info["rows"] = lines[1:]
    return raw, info


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    raw_a, a = describe(sys.argv[1])
    raw_b, b = describe(sys.argv[2])

    for name, info in (("original", a), ("regenerated", b)):
        print("{:<12} {}".format(name, info["path"]))
        print("  SHA-256    {}".format(info["sha256"]))
        print("  size       {:,} bytes   encoding {}   newlines {}   BOM {}   final newline {}".format(
            info["bytes"], info["encoding"], info["newlines"], "yes" if info["bom"] else "no",
            "yes" if info["ends_with_newline"] else "no"))
        print("  header     {!r}   data rows {:,}".format(info["header"], len(info["rows"])))

    problems = []
    for name, info in (("original", a), ("regenerated", b)):
        if info["header"] != EXPECTED_HEADER:
            problems.append("{}: header is {!r}, expected {!r}".format(
                name, info["header"], EXPECTED_HEADER))
        if len(info["rows"]) != EXPECTED_ROWS:
            problems.append("{}: {:,} data rows, expected {:,}".format(
                name, len(info["rows"]), EXPECTED_ROWS))
        bad = next((r for r in info["rows"] if r.count(",") != 1
                    or r.split(",")[1] not in LABELS), None)
        if bad is not None:
            problems.append("{}: malformed row {!r}".format(name, bad))

    print()
    if raw_a == raw_b:
        print("Checks: same row count, same header and column order, same row order, same values,")
        print("same encoding and newlines -- the two files are identical byte for byte.")
        print("RESULT: IDENTICAL (SHA-256 {})".format(a["sha256"]))
        for p in problems:
            print("  note: {}".format(p))
        return 0

    print("RESULT: DIFFERENT")
    for p in problems:
        print("  - {}".format(p))
    if a["header"] != b["header"]:
        print("  - headers differ: {!r} vs {!r}".format(a["header"], b["header"]))
    if len(a["rows"]) != len(b["rows"]):
        print("  - row counts differ: {:,} vs {:,}".format(len(a["rows"]), len(b["rows"])))
    for key in ("encoding", "newlines", "bom", "ends_with_newline"):
        if a[key] != b[key]:
            print("  - {} differs: {} vs {}".format(key, a[key], b[key]))
    first = next((i for i, (x, y) in enumerate(zip(a["rows"], b["rows"])) if x != y), None)
    if first is not None:
        x, y = a["rows"][first], b["rows"][first]
        kind = "row order (frame column)" if x.split(",")[0] != y.split(",")[0] else "label value"
        n_diff = sum(1 for x2, y2 in zip(a["rows"], b["rows"]) if x2 != y2)
        print("  - first difference at data row {:,} ({}): {!r} vs {!r}; {:,} rows differ".format(
            first + 1, kind, x, y, n_diff))
    return 1


if __name__ == "__main__":
    sys.exit(main())
