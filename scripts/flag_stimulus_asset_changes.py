#!/usr/bin/env python3
"""Flag (do NOT auto-fix) drift in the bridge2ai-redcap source that feeds the task
registry. Runs per redcap2rs release (in the pipeline_trigger auto-PR flow).

Why this exists: redcap2rs converts the RedCap *data dictionary* only, so changes under
bridge2ai-redcap `docs/` -- task instructions/descriptions, protocols, and the stimulus
images/cards -- are NOT surfaced by the auto-PR. Those docs feed our task registry, banks,
and pinned `stimulus_asset` URLs, so they need their own drift alert.

Design decisions:
  * stimulus_asset URLs are pinned to a commit ON PURPOSE (immutable; shows the image
    contemporaneous with that release's data). We do NOT auto-bump -- re-pinning could show a
    different image than the participant saw. Re-pinning / re-transcription is a deliberate,
    human, version-scoped step.
  * Transcribed cards (identifying-pictures words, sound tokens) need OCR / a vision pass to
    re-read, so this script RAISES A FLAG; it never transcribes or writes the registry.

Checks per release:
  1. LINK-ROT: every registry-referenced image still resolves at its pinned commit (full
     expected set, not just "at least one").
  2. IMAGE DRIFT (per task): pinned commit vs new release -- added/removed/CHANGED (blob SHA);
     a change on a text_bank task -> "needs re-transcription + new-pin decision".
  3. DOCS DRIFT (docs/ tree, blob-SHA diff pinned-baseline vs new release) -- text/protocol
     changes listed, other/image changes counted.

Exit codes: 0 = no action needed; 1 = drift/link-rot flagged; 2 = the check itself failed to
run (e.g. API/network error). A report is ALWAYS written so the caller never silently no-ops.
stdlib only; honors GITHUB_TOKEN.
"""
import argparse, json, os, re, sys, collections, urllib.request, urllib.error
from urllib.parse import quote
from pathlib import Path

REPO = "eipm/bridge2ai-redcap"
API = "https://api.github.com"
TEXT_EXT = {".md", ".txt", ".json", ".csv", ".yaml", ".yml"}
PLACEHOLDER = re.compile(r"\{[^}]*\}")   # any {i}, {i:02d}, {n}, {page}, ...

def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

_dir_cache = {}
def dir_blobs(path, commit):
    """{filename: blob_sha} directly under `path` at `commit`; None if the dir is gone."""
    key = (path, commit)
    if key in _dir_cache:
        return _dir_cache[key]
    try:
        items = _get(f"{API}/repos/{REPO}/contents/{quote(path)}?ref={commit}")
        out = {it["name"]: it["sha"] for it in items if it.get("type") == "file"}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            out = None
        else:
            raise
    _dir_cache[key] = out
    return out

def tree_blobs(commit, prefix="docs/"):
    data = _get(f"{API}/repos/{REPO}/git/trees/{commit}?recursive=1")
    blobs = {t["path"]: t["sha"] for t in data.get("tree", [])
             if t.get("type") == "blob" and t["path"].startswith(prefix)}
    return blobs, data.get("truncated", False)

def targets(pr):
    """List of (dir, matcher, is_regex). asset_map -> one entry per file in ITS OWN dir;
    templated -> one regex entry; fixed -> one exact entry."""
    if pr.get("asset_map"):
        out = []
        for p in pr["asset_map"].values():
            d, n = p.rsplit("/", 1)
            out.append((d, n, False))
        return out
    p = pr.get("asset_path")
    if not p:
        return []
    d, base = p.rsplit("/", 1)
    if "{" in base:
        parts = PLACEHOLDER.split(base)
        return [(d, re.compile("^" + r"\d+".join(re.escape(x) for x in parts) + "$"), True)]
    return [(d, base, False)]

def actual_shas(pr, commit):
    """{full_path: blob_sha} of the task's images actually present at `commit`."""
    out = {}
    for d, m, is_rx in targets(pr):
        blobs = dir_blobs(d, commit)
        if not blobs:
            continue
        if is_rx:
            for name, sha in blobs.items():
                if m.match(name):
                    out[f"{d}/{name}"] = sha
        elif m in blobs:
            out[f"{d}/{m}"] = blobs[m]
    return out

def expected_count(task, pr, bank_dir):
    tb = pr.get("text_bank")
    if tb:
        return len(json.loads((bank_dir / f"{tb}.json").read_text()).get("words", []))
    if isinstance(pr.get("asset_count"), int):
        return pr["asset_count"]
    bank = pr.get("bank")
    if bank:
        b = json.loads((bank_dir / f"{bank}.json").read_text())
        items = b.get("sentences") or b.get("words") or []
        return len(items) or None
    return None

def expected_paths(task, pr, bank_dir):
    """The full set of paths the registry EXPECTS to exist, or None if not enumerable."""
    if pr.get("asset_map"):
        return set(pr["asset_map"].values())
    p = pr.get("asset_path")
    if not p:
        return set()
    if "{" not in p:
        return {p}
    n = expected_count(task, pr, bank_dir)
    if not n:
        return None                       # templated but count unknown
    # fill the placeholder for 1..n (i and n kwargs both supplied; extras ignored)
    return {p.format(i=k, n=k) for k in range(1, n + 1)}

def run(args):
    registry = json.loads(Path(args.registry).read_text())
    bank_dir = Path(args.registry).parent
    new_ref = None if args.link_rot_only else (args.ref or _get(f"{API}/repos/{REPO}")["default_branch"])
    new_sha = _get(f"{API}/repos/{REPO}/commits/{new_ref}")["sha"] if new_ref else None

    rot, drift = [], []
    pins = collections.Counter()
    for tid, task in registry["tasks"].items():
        pr = task.get("prompt_ref") or {}
        if not pr.get("asset_repo") or not targets(pr):
            continue
        pinned = pr.get("asset_commit")
        if pinned:
            pins[pinned] += 1
        transcribed = bool(pr.get("text_bank"))

        old = actual_shas(pr, pinned) if pinned else {}
        exp = expected_paths(task, pr, bank_dir)
        if exp is None:                                   # can't enumerate -> weak check
            if not old:
                rot.append(f"[{tid}] no images resolve at pinned {(pinned or '?')[:12]}")
        else:
            missing = sorted(exp - set(old))
            if missing:
                rot.append(f"[{tid}] {len(missing)} pinned image(s) missing at "
                           f"{(pinned or '?')[:12]}: {missing[:5]}")

        if new_sha:
            new = actual_shas(pr, new_sha)
            a = sorted(set(new) - set(old)); r = sorted(set(old) - set(new))
            c = sorted(p for p in set(old) & set(new) if old[p] != new[p])
            if a or r or c:
                tag = "NEEDS RE-TRANSCRIPTION (OCR/vision) + new-pin decision" if transcribed \
                      else "images changed; verify + decide on a new pin"
                drift.append(f"[{tid}] {tag}"
                             + (f" | added={[os.path.basename(x) for x in a]}" if a else "")
                             + (f" | removed={[os.path.basename(x) for x in r]}" if r else "")
                             + (f" | changed={[os.path.basename(x) for x in c]}" if c else ""))

    # DOCS DRIFT: whole docs/ tree, pinned baseline vs new release
    docs_text, docs_other_n, truncated, baseline = [], 0, False, None
    if new_sha:
        baseline = args.baseline or (pins.most_common(1)[0][0] if pins else None)
        if baseline:
            base, t1 = tree_blobs(baseline, args.docs_prefix)
            new, t2 = tree_blobs(new_sha, args.docs_prefix)
            truncated = t1 or t2
            for p in sorted(set(base) | set(new)):
                b, n = base.get(p), new.get(p)
                if b == n:
                    continue
                kind = "added" if b is None else "removed" if n is None else "changed"
                if os.path.splitext(p)[1].lower() in TEXT_EXT:
                    docs_text.append(f"{kind}: {p}")
                else:
                    docs_other_n += 1

    lines = ["# bridge2ai-redcap drift flags", ""]
    if new_sha:
        lines.append(f"checked vs {REPO}@{new_ref} (`{new_sha[:12]}`)"
                     + (f"; docs baseline `{baseline[:12]}`" if baseline else "; docs baseline: none"))
    lines.append(f"link-rot: {len(rot) or 'none'}; image drift: {len(drift) or 'none'} task(s); "
                 f"docs text/protocol changes: {len(docs_text) or 'none'}; "
                 f"docs other/image changes: {docs_other_n or 'none'}")
    if truncated:
        lines.append("WARNING: docs tree listing truncated by the API; results may be incomplete.")
    lines.append("")
    if rot:
        lines += ["## LINK-ROT (pinned images not resolving):"] + [f"- {x}" for x in rot] + [""]
    if drift:
        lines += ["## IMAGE DRIFT (needs human action):"] + [f"- {x}" for x in drift] + [""]
    if docs_text:
        lines += ["## DOCS DRIFT -- text/protocol changed (not surfaced by redcap2rs; review):"] \
                 + [f"- {x}" for x in docs_text] + [""]
    if docs_other_n:
        lines += [f"## DOCS DRIFT -- {docs_other_n} other/image file(s) changed under "
                  f"{args.docs_prefix} (referenced ones flagged under IMAGE DRIFT; the rest may "
                  "be new tasks/cards to add)", ""]
    if not (rot or drift or docs_text or docs_other_n):
        lines.append("No action needed.")
    return "\n".join(lines), bool(rot or drift or docs_text or docs_other_n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--ref", default=None, help="new bridge2ai-redcap release (branch/tag/sha); default=default branch")
    ap.add_argument("--baseline", default=None, help="docs baseline commit; default = the registry's pinned commit")
    ap.add_argument("--link-rot-only", action="store_true")
    ap.add_argument("--docs-prefix", default="docs/")
    ap.add_argument("--report", default="stimulus_asset_flags.md")
    args = ap.parse_args()
    try:
        report, flagged = run(args)
        code = 1 if flagged else 0
    except Exception as e:                 # never silently no-op: emit an error report, exit 2
        report = ("# bridge2ai-redcap drift flags\n\n"
                  f"ERROR: the drift check failed to run: {type(e).__name__}: {e}\n"
                  "This is NOT a clean result -- rerun / investigate before trusting it.")
        code = 2
    Path(args.report).write_text(report)
    print(report)
    sys.exit(code)

if __name__ == "__main__":
    main()
