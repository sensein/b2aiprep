#!/usr/bin/env python3
"""Flag (do NOT auto-fix) drift in the bridge2ai-redcap source that feeds the task
registry. Runs per redcap2rs release (in the pipeline_trigger auto-PR flow).

Why this exists: redcap2rs converts the RedCap *data dictionary* only, so changes under
bridge2ai-redcap `docs/` -- task instructions/descriptions, protocols, and the stimulus
images/cards -- are NOT surfaced by the #331 auto-PR. Those docs feed our task registry,
banks, and pinned `stimulus_asset` URLs, so they need their own drift alert.

Design decisions:
  * stimulus_asset URLs are pinned to a commit ON PURPOSE (immutable; shows the image
    contemporaneous with that release's data). We do NOT auto-bump the pin -- re-pinning to
    a newer commit could show a different image than the participant saw. Re-pinning /
    re-transcription is a deliberate, human, version-scoped step.
  * Transcribed cards (identifying-pictures words, sound tokens) need OCR / a vision pass to
    re-read, so this script RAISES A FLAG; it never transcribes or writes the registry.

Three checks (all per release):
  1. LINK-ROT: every registry-referenced image still resolves at its pinned commit.
  2. IMAGE DRIFT (per task): pinned commit vs new release -- added/removed/CHANGED (blob SHA);
     changes on a text_bank task -> "needs re-transcription + new-pin decision".
  3. DOCS DRIFT (tree-wide): every file under docs/ that changed between the registry's
     pinned commit and the new release -- text/protocol changes listed explicitly (they may
     require registry/bank/instruction updates that redcap2rs won't catch).

Exit non-zero if anything is flagged. stdlib only; honors GITHUB_TOKEN.
"""
import argparse, json, os, re, sys, collections, urllib.request, urllib.error
from urllib.parse import quote
from pathlib import Path

REPO = "eipm/bridge2ai-redcap"
API = "https://api.github.com"
TEXT_EXT = {".md", ".txt", ".json", ".csv", ".yaml", ".yml"}

def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

_dir_cache = {}
def dir_blobs(path, commit):
    key = (path, commit)
    if key in _dir_cache: return _dir_cache[key]
    try:
        items = _get(f"{API}/repos/{REPO}/contents/{quote(path)}?ref={commit}")
        out = {it["name"]: it["sha"] for it in items if it.get("type") == "file"}
    except urllib.error.HTTPError as e:
        if e.code == 404: out = None
        else: raise
    _dir_cache[key] = out
    return out

def tree_blobs(commit, prefix="docs/"):
    """{path: blob_sha} for every file under `prefix` at `commit`; (blobs, truncated)."""
    data = _get(f"{API}/repos/{REPO}/git/trees/{commit}?recursive=1")
    blobs = {t["path"]: t["sha"] for t in data.get("tree", [])
             if t.get("type") == "blob" and t["path"].startswith(prefix)}
    return blobs, data.get("truncated", False)

def image_dir_and_pattern(pr):
    if pr.get("asset_map"):
        paths = list(pr["asset_map"].values())
        return paths[0].rsplit("/", 1)[0], None, {p.rsplit("/", 1)[1] for p in paths}
    p = pr.get("asset_path")
    if not p: return None, None, set()
    d, base = p.rsplit("/", 1)
    if "{" in base:
        parts = re.split(r"\{[in][^}]*\}", base)
        return d, re.compile("^" + r"\d+".join(re.escape(x) for x in parts) + "$"), None
    return d, None, {base}

def selected(blobs, rx, explicit):
    if blobs is None: return None
    if explicit is not None: return {n: s for n, s in blobs.items() if n in explicit}
    return {n: s for n, s in blobs.items() if rx.match(n)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--ref", default=None, help="new bridge2ai-redcap release (branch/tag/sha); default=default branch")
    ap.add_argument("--baseline", default=None, help="docs baseline commit; default = the registry's pinned commit")
    ap.add_argument("--link-rot-only", action="store_true")
    ap.add_argument("--docs-prefix", default="docs/")
    ap.add_argument("--report", default="stimulus_asset_flags.md")
    args = ap.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    new_ref = None if args.link_rot_only else (args.ref or _get(f"{API}/repos/{REPO}")["default_branch"])
    new_sha = _get(f"{API}/repos/{REPO}/commits/{new_ref}")["sha"] if new_ref else None

    rot, drift = [], []
    pins = collections.Counter()
    for tid, task in registry["tasks"].items():
        pr = task.get("prompt_ref") or {}
        if not pr.get("asset_repo"): continue
        pinned = pr.get("asset_commit"); pins[pinned] += 1
        transcribed = bool(pr.get("text_bank"))
        d, rx, explicit = image_dir_and_pattern(pr)
        if not d: continue
        old = selected(dir_blobs(d, pinned), rx, explicit)
        if old is None or (explicit and not explicit.issubset(set(old or {}))) or (rx and not old):
            rot.append(f"[{tid}] pinned images do not resolve at {pinned[:12]} ({d})"); old = old or {}
        if new_sha:
            new = selected(dir_blobs(d, new_sha), rx, explicit)
            if new is None:
                drift.append(f"[{tid}] asset dir removed at {new_sha[:12]}: {d}"); continue
            a = sorted(set(new)-set(old)); r = sorted(set(old)-set(new))
            c = sorted(n for n in set(old)&set(new) if old[n] != new[n])
            if a or r or c:
                tag = "NEEDS RE-TRANSCRIPTION (OCR/vision) + new-pin decision" if transcribed \
                      else "images changed; verify + decide on a new pin"
                drift.append(f"[{tid}] {tag}" + (f" | added={a}" if a else "") +
                             (f" | removed={r}" if r else "") + (f" | changed={c}" if c else ""))

    # DOCS DRIFT: whole docs/ tree, pinned baseline vs new release
    docs_text, docs_img_n, truncated = [], 0, False
    if new_sha:
        baseline = args.baseline or (pins.most_common(1)[0][0] if pins else None)
        if baseline:
            base, t1 = tree_blobs(baseline, args.docs_prefix)
            new, t2 = tree_blobs(new_sha, args.docs_prefix)
            truncated = t1 or t2
            paths = set(base) | set(new)
            for p in sorted(paths):
                b, n = base.get(p), new.get(p)
                if b == n: continue
                kind = "added" if b is None else "removed" if n is None else "changed"
                if os.path.splitext(p)[1].lower() in TEXT_EXT:
                    docs_text.append(f"{kind}: {p}")
                else:
                    docs_img_n += 1

    lines = ["# bridge2ai-redcap drift flags", ""]
    if new_sha: lines.append(f"checked vs {REPO}@{new_ref} (`{new_sha[:12]}`); docs baseline `{(args.baseline or (pins.most_common(1)[0][0] if pins else ''))[:12]}`")
    lines.append(f"link-rot: {len(rot) or 'none'}; image drift: {len(drift) or 'none'} task(s); "
                 f"docs text/protocol changes: {len(docs_text) or 'none'}; docs image/other changes: {docs_img_n or 'none'}")
    if truncated: lines.append("WARNING: docs tree listing was truncated by the API; results may be incomplete.")
    lines.append("")
    if rot:   lines += ["## LINK-ROT (pinned images not resolving):"] + [f"- {x}" for x in rot] + [""]
    if drift: lines += ["## IMAGE DRIFT (needs human action):"] + [f"- {x}" for x in drift] + [""]
    if docs_text:
        lines += ["## DOCS DRIFT -- text/protocol changed (not surfaced by redcap2rs; review):"] + [f"- {x}" for x in docs_text] + [""]
    if docs_img_n:
        lines += [f"## DOCS DRIFT -- {docs_img_n} image/other file(s) changed under {args.docs_prefix} "
                  "(referenced ones are flagged under IMAGE DRIFT; the rest may be new tasks/cards to add)", ""]
    if not (rot or drift or docs_text or docs_img_n):
        lines.append("No action needed.")
    report = "\n".join(lines)
    Path(args.report).write_text(report); print(report)
    sys.exit(1 if (rot or drift or docs_text or docs_img_n) else 0)

if __name__ == "__main__":
    main()
