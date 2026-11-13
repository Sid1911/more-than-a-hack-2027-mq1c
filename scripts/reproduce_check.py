"""Clean reproduction test: fresh clone, clean environment, regenerate, compare byte for byte.

    python scripts/reproduce_check.py --data-dir PATH\\TO\\bundle --original PATH\\TO\\MQ1C.csv
    python scripts/reproduce_check.py --repo https://github.com/<user>/<repo> --data-dir ... --original ...

Steps, each one printed as it runs:
  1. clone the repository into a new empty folder (default source: this repository's own git
     folder; pass --repo with the public URL to test exactly what others will download);
  2. make a new Python environment there and install requirements.txt with no pip cache;
  3. run the documented command, scripts/predict.py, with --no-cache so the tracking file is
     parsed from scratch rather than read from an earlier run's cache;
  4. compare the regenerated file with --original byte for byte (scripts/verify_submission.py).

Exits 0 only if the files are identical. Leaves the clone in place and writes a short report,
reproduction_report.txt, next to it. Needs git on the PATH and Python 3.9-3.12.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, log, cwd=None):
    shown = " ".join('"{}"'.format(c) if " " in c else c for c in cmd)
    print("\n$ " + shown, flush=True)
    log.append("$ " + shown)
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    print(proc.stdout, end="", flush=True)
    log.append(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit("step failed (exit code {}): {}".format(proc.returncode, shown))
    return proc.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HERE,
                    help="what to clone: the public URL, or a local repository (default: this one)")
    ap.add_argument("--data-dir", required=True, help="the separately supplied data bundle")
    ap.add_argument("--original", required=True, help="our original submission file")
    ap.add_argument("--python", default=sys.executable,
                    help="Python used to create the clean environment (default: this one)")
    ap.add_argument("--workdir", help="where to clone (default: a new temporary folder)")
    args = ap.parse_args()

    if shutil.which("git") is None:
        sys.exit("git is not on the PATH: install Git for Windows, then run this again.")
    original = os.path.abspath(args.original)
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isfile(original):
        sys.exit("original submission not found: " + original)

    base = args.workdir or tempfile.mkdtemp(prefix="mtah-repro-")
    os.makedirs(base, exist_ok=True)
    clone = os.path.join(base, "repo")
    if os.path.exists(clone):
        sys.exit("{} already exists: choose an empty --workdir".format(clone))
    log = ["clean reproduction test, {}".format(datetime.datetime.now().isoformat(timespec="seconds")),
           "repository: {}".format(args.repo), "data: {}".format(data_dir),
           "original: {}".format(original)]

    print("1. fresh clone", flush=True)
    run(["git", "clone", "--quiet", args.repo, clone], log)
    commit = run(["git", "-C", clone, "rev-parse", "HEAD"], log).strip()
    status = run(["git", "-C", clone, "status", "--porcelain", "--ignored"], log).strip()
    if status:
        sys.exit("the fresh clone is not clean:\n" + status)

    print("\n2. clean Python environment", flush=True)
    venv = os.path.join(base, "venv")
    run([args.python, "-m", "venv", venv], log)
    py = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
    run([py, "--version"], log)
    run([py, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "-q",
         "-r", os.path.join(clone, "requirements.txt")], log)
    run([py, "-m", "pip", "freeze", "--disable-pip-version-check"], log)

    print("\n3. the documented command", flush=True)
    run([py, os.path.join(clone, "scripts", "predict.py"), "--data-dir", data_dir, "--no-cache"], log)
    team = open(os.path.join(clone, "team.txt")).read().strip()
    regenerated = os.path.join(clone, "submission", team + ".csv")

    print("\n4. byte-for-byte comparison", flush=True)
    proc = subprocess.run([py, os.path.join(clone, "scripts", "verify_submission.py"), original,
                           regenerated], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    print(proc.stdout, flush=True)
    log.append(proc.stdout)

    report = os.path.join(base, "reproduction_report.txt")
    with open(report, "w") as handle:
        handle.write("\n".join(log))
        handle.write("\ncommit {}\nresult: {}\n".format(
            commit, "PASS" if proc.returncode == 0 else "FAIL"))
    print("commit tested: {}".format(commit))
    print("report: {}".format(report))
    print("REPRODUCTION {}".format("PASSED" if proc.returncode == 0 else "FAILED"))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
