"""Audit the repository before making it public: no competition data anywhere in it.

    python scripts/audit_repo.py

Checks every object git would publish -- all tracked files (hidden ones included) and every
file in every commit of every branch and tag, not only the current one -- for:

  * names of competition files (BAR-ATH..., *_atd.*, *homography*, *smart_tagging*, videos,
    caches, submissions) or anything under data/ other than data/README.md;
  * contents that look like competition data whatever the file is called: tracking lines
    ("12345:x,y,s;...:bx,by,bz"), Smart Tagging XML (<instance> ... <code>), a submission or
    ground-truth table (frame,label with thousands of rows), numpy archives, video headers;
  * any file over 1 MB.

It also lists files that are in the working folder but ignored by git (the venv, the data, the
cache, the submission): those are not published. Exits 0 only if nothing was found.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 1000000
NAME_PATTERNS = [r"BAR-ATH", r"_atd\.", r"homography", r"smart_tagging", r"\.mp4$", r"\.mov$",
                 r"\.npz$", r"\.npy$", r"^submission/", r"^data/(?!README\.md$)", r"\.parquet$",
                 r"\.pkl$", r"cache"]
ATD_LINE = re.compile(rb"^\d+:[-\d.eE,NaN]+;[-\d.eE,NaN;]+:[-\d.eE,NaN]+", re.M)
TAG_XML = re.compile(rb"<instance>.*?<code>", re.S)
FRAME_LABEL = re.compile(rb"^\d+,[HADX]\r?$", re.M)


def git(*args):
    return subprocess.run(["git", "-C", ROOT] + list(args), stdout=subprocess.PIPE,
                          check=True).stdout


def main():
    findings = []

    tracked = git("ls-files", "-z").decode().split("\0")
    tracked = [t for t in tracked if t]
    print("tracked files ({}), hidden ones included:".format(len(tracked)))
    for t in tracked:
        print("   " + t)

    # every blob reachable from any ref, with the path it was committed under
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    blobs = {}
    for line in objects:
        sha, _, path = line.partition(" ")
        if path:
            blobs.setdefault(sha, set()).add(path)
    sizes = {}
    if blobs:
        out = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch-check"],
                             input="\n".join(blobs).encode(), stdout=subprocess.PIPE,
                             check=True).stdout.decode().splitlines()
        for line in out:
            sha, kind, size = line.split()
            if kind == "blob":
                sizes[sha] = int(size)
    history_paths = sorted({p for sha in sizes for p in blobs[sha]})
    commits = git("rev-list", "--all", "--count").decode().strip()
    print("\nhistory: {} commits, {} file versions, {} distinct paths".format(
        commits, len(sizes), len(history_paths)))

    for path in history_paths:
        for pat in NAME_PATTERNS:
            if re.search(pat, path, re.I):
                findings.append("name looks like competition data or a generated file: " + path)
                break

    for sha, size in sizes.items():
        where = ", ".join(sorted(blobs[sha]))
        if size > MAX_BYTES:
            findings.append("file over 1 MB in history: {} ({:,} bytes)".format(where, size))
            continue
        body = git("cat-file", "-p", sha)
        if ATD_LINE.search(body):
            findings.append("tracking-like lines in: " + where)
        if TAG_XML.search(body) and body.count(b"<instance>") >= 20:
            findings.append("Smart-Tagging-like XML in: " + where)
        if len(FRAME_LABEL.findall(body)) > 1000:
            findings.append("submission/ground-truth-like table in: " + where)
        if body[:6] == b"\x93NUMPY" or body[:2] == b"PK" and b".npy" in body[:200]:
            findings.append("numpy data in: " + where)
        if body[4:8] in (b"ftyp", b"moov"):
            findings.append("video in: " + where)

    ignored = git("status", "--porcelain", "--ignored", "-z").decode().split("\0")
    ignored = [e[3:] for e in ignored if e.startswith("!! ")]
    untracked = git("status", "--porcelain", "-z").decode().split("\0")
    untracked = [e[3:] for e in untracked if e.startswith("?? ")]
    print("\nin the folder but ignored by git, so never published: {}".format(
        ", ".join(ignored) if ignored else "nothing"))
    if untracked:
        print("NOT ignored and not committed yet (check before adding): " + ", ".join(untracked))
        for u in untracked:
            for pat in NAME_PATTERNS:
                if re.search(pat, u, re.I):
                    findings.append("unignored file looks like competition data: " + u)
                    break

    print()
    if findings:
        print("AUDIT FAILED -- do not publish:")
        for f in findings:
            print("   - " + f)
        return 1
    print("AUDIT PASSED: no competition data, cache, submission or large file anywhere in the")
    print("tracked files or the git history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
