"""Release scan: fails on any local path, e-mail address, listed term, em dash or internal file.

  python tools/scan_package.py [--terms FILE] [--git --expect-identity "Name <address>"]

Rules, applied to every text file in the package (py, md, json, csv, cff, txt, ps1, LICENSE,
.gitignore) and to every file name:
  local paths   a drive letter followed by a backslash, a macOS home directory, a synced
                cloud folder name
  e-mail        any address; none may appear in a file
  terms         each line of the optional --terms file, matched case-insensitively (the release
                check uses a private list of names that must not appear in a public file)
  em dash       the character U+2014 anywhere
  internal      files under paper/ or briefs/, dell_kit, image or document files, pickles,
                audit sheets, bytecode, and any file type outside the release set
With --git the repository must hold exactly one commit whose author and committer are the
expected identity and whose message has no trailer lines.
"""
import argparse
import os
import re
import subprocess
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "archive", ".venv"}
TEXT_EXT = {".py", ".md", ".json", ".csv", ".cff", ".txt", ".ps1"}
TEXT_NAMES = {"LICENSE", ".gitignore"}
BINARY_EXT = {".npz"}
FORBIDDEN_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".pdf", ".pkl", ".pyc", ".docx", ".xlsx", ".xls",
                 ".pptx", ".onnx", ".pt", ".pth", ".zip"}
PATTERNS = [
    ("local path", re.compile(r"[A-Za-z]:\\")),
    ("local path", re.compile("/" + "Users" + "/")),
    ("local path", re.compile("one" + "drive", re.I)),
    ("e-mail", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("em dash", re.compile(chr(0x2014))),
]


def scan_text(rel, text, terms, hits):
    for n, line in enumerate(text.splitlines(), 1):
        for rule, rx in PATTERNS:
            for m in rx.finditer(line):
                hits.append(f"{rel}:{n}: {rule}: ...{line[max(0, m.start() - 40):m.end() + 40]}...")
        low = line.lower()
        for t in terms:
            i = low.find(t)
            if i >= 0:
                hits.append(f"{rel}:{n}: listed term '{t}': ...{line[max(0, i - 40):i + len(t) + 40]}...")


def scan_files(terms):
    hits, scanned, binaries = [], 0, 0
    for root, dirs, files in os.walk(PKG):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for d in dirs:
            if d == "__pycache__":
                hits.append(f"{os.path.relpath(os.path.join(root, d), PKG)}: internal: bytecode folder")
        for f in sorted(files):
            path = os.path.join(root, f)
            rel = os.path.relpath(path, PKG).replace(os.sep, "/")
            ext = os.path.splitext(f)[1].lower()
            parts = rel.lower().split("/")
            if parts[0] in ("paper", "briefs") or any("dell_kit" in p for p in parts):
                hits.append(f"{rel}: internal: file from an internal folder")
            if "audit" in f.lower() and ext not in (".json", ".md"):
                hits.append(f"{rel}: internal: audit sheet")
            for t in terms:
                if t in rel.lower():
                    hits.append(f"{rel}: listed term '{t}' in the file name")
            if ext in FORBIDDEN_EXT:
                hits.append(f"{rel}: internal: {ext} file")
                continue
            if ext in BINARY_EXT:
                binaries += 1
                continue
            if ext not in TEXT_EXT and f not in TEXT_NAMES:
                hits.append(f"{rel}: internal: unexpected file type")
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            try:
                text = data.decode("utf8")
            except UnicodeDecodeError:
                hits.append(f"{rel}: not valid UTF-8")
                continue
            scan_text(rel, text, terms, hits)
            scanned += 1
    return hits, scanned, binaries


def git(*args):
    return subprocess.run(["git", "-C", PKG, *args], capture_output=True, text=True, check=True).stdout


def scan_git(identity, hits):
    count = int(git("rev-list", "--count", "HEAD").strip())
    if count != 1:
        hits.append(f"git: {count} commits, expected 1")
    for fmt, who in (("%an <%ae>", "author"), ("%cn <%ce>", "committer")):
        value = git("log", "-1", f"--format={fmt}").strip()
        if value != identity:
            hits.append(f"git: {who} is '{value}', expected the given identity")
    if git("log", "-1", "--format=%(trailers)").strip():
        hits.append("git: the commit message has trailer lines")
    body = git("log", "-1", "--format=%B")
    terms_in_msg = [line for line in body.splitlines() if re.match(r"^[\w-]+: ", line)]
    if terms_in_msg:
        hits.append(f"git: message lines that look like trailers: {terms_in_msg}")
    return body.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", help="file with one term per line, matched case-insensitively")
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--expect-identity")
    args = ap.parse_args()
    terms = []
    if args.terms:
        with open(args.terms, encoding="utf8") as f:
            terms = [t.strip().lower() for t in f if t.strip() and not t.startswith("#")]
    hits, scanned, binaries = scan_files(terms)
    message = None
    if args.git:
        if not args.expect_identity:
            raise SystemExit("--git needs --expect-identity")
        message = scan_git(args.expect_identity, hits)
    print(f"text files scanned: {scanned}; npz files checked by name: {binaries}; listed terms: {len(terms)}")
    if message is not None:
        print(f"git: commit message: {message}")
    for h in hits:
        print("HIT", h)
    print(f"hits: {len(hits)}")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
