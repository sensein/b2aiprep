"""Prepare (and, separately, submit) a new-version DOI for a Synapse entity by
cloning the metadata of its existing DOI -- so the long author/creator list,
title, resourceType, etc. carry over and only the version changes.

Two-step, review-gated workflow:

  1) PREPARE (read-only; default):
       python prepare_doi.py --view syn74341293 --to-version 2
     Fetches the existing DOI (current, or --from-version N), bumps objectVersion
     to the already-created snapshot version, strips server/identity fields, writes
     the payload to --out (default prepared_doi.json), and prints a summary. Nothing
     is minted. Review (and optionally edit) the JSON.

  2) SUBMIT (mints the DOI; must point at the reviewed file):
       python prepare_doi.py --submit --payload prepared_doi.json

Auth: SAGE_PAT or SYNAPSE_PAT in the environment, else ~/.synapseConfig.
"""
import os
import sys
import json
import time
import argparse
import urllib.request

import synapseclient

# Server-assigned / old-DOI-identity fields -- never copy these into the new DOI.
_STRIP = {"associationId", "associatedBy", "associatedOn", "etag", "doiUri", "doiUrl",
          "doiId", "createdBy", "createdOn", "updatedBy", "updatedOn"}


def login():
    syn = synapseclient.Synapse()
    tok = os.environ.get("SAGE_PAT") or os.environ.get("SYNAPSE_PAT")
    syn.login(authToken=tok) if tok else syn.login()
    print(f"LOGIN OK as {syn.getUserProfile().get('userName')}", file=sys.stderr)
    return syn


def fetch_existing_doi(syn, view, from_version):
    # NOTE: the GET /doi query param for the version is `version` (the Doi *object*
    # field is `objectVersion`). A version-pinned DOI is invisible without it.
    q = f"/doi?id={view}&type=ENTITY"
    if from_version is not None:
        q += f"&version={from_version}"
    return syn.restGET(q)


def entity_versions(syn, view):
    try:
        return {v["versionNumber"] for v in syn.restGET(f"/entity/{view}/version")["results"]}
    except Exception:
        return set()


def build_new_payload(existing, to_version):
    doi = {k: v for k, v in existing.items() if k not in _STRIP}
    doi["objectVersion"] = to_version
    doi.setdefault("concreteType", "org.sagebionetworks.repo.model.doi.v2.Doi")
    return doi


def fetch_datacite_creators(doi):
    """Creator names from a DataCite record (correct UTF-8; Synapse GET mangles them)."""
    with urllib.request.urlopen(f"https://api.datacite.org/dois/{doi}", timeout=30) as r:
        data = json.load(r)
    return [c.get("name") for c in data["data"]["attributes"].get("creators", [])]


def apply_datacite_names(payload, dc_names):
    """Replace the (Synapse-mangled) creatorNames containing '?' with the correctly
    encoded DataCite names, matched by position. Aborts if the two lists don't line
    up (length, or any ASCII name disagrees -> different ordering). Returns the
    (old, new) pairs that were corrected."""
    cr = payload.get("creators", [])
    if len(cr) != len(dc_names):
        raise SystemExit(f"creator-count mismatch: payload {len(cr)} vs DataCite {len(dc_names)} "
                         "-- cannot align by position.")
    # Compare trimmed: whitespace-only differences (stray trailing spaces exist in
    # the stored names) are not real misalignments.
    mism = [(i, cr[i]["creatorName"], dc_names[i]) for i in range(len(cr))
            if "?" not in cr[i]["creatorName"]
            and cr[i]["creatorName"].strip() != (dc_names[i] or "").strip()]
    if mism:
        raise SystemExit(f"DataCite/Synapse creator order disagrees on {len(mism)} ASCII name(s) "
                         f"(e.g. {mism[:2]}); refusing to remap.")
    fixed = []
    for i, c in enumerate(cr):
        if "?" in c["creatorName"]:
            fixed.append((c["creatorName"], dc_names[i]))
            c["creatorName"] = dc_names[i]
    return fixed


def _title(doi):
    if doi.get("title"):
        return doi["title"]
    t = doi.get("titles")
    if t:
        return t[0].get("title") if isinstance(t[0], dict) else t[0]
    return None


def summarize(doi, existing):
    creators = doi.get("creators", [])
    print("\n================ PREPARED DOI (REVIEW) ================")
    print(f"  objectId        : {doi.get('objectId')}")
    print(f"  objectType      : {doi.get('objectType')}")
    print(f"  objectVersion   : {existing.get('objectVersion')}  ->  {doi.get('objectVersion')}")
    print(f"  title           : {_title(doi)}")
    print(f"  publisher       : {doi.get('publisher')}")
    print(f"  publicationYear : {doi.get('publicationYear')}")
    print(f"  resourceType    : {doi.get('resourceType')}")
    print(f"  creators        : {len(creators)} carried over")
    if creators:
        show = [c.get("creatorName", c) for c in creators[:3]]
        print(f"      e.g. {show}{' ...' if len(creators) > 3 else ''}")
    other = sorted(set(doi) - {"objectId", "objectType", "objectVersion", "title", "titles",
                               "publisher", "publicationYear", "resourceType", "creators", "concreteType"})
    if other:
        print(f"  other metadata carried over: {other}")
    print("======================================================\n")


def submit(syn, payload):
    print("Submitting DOI mint job ...", file=sys.stderr)
    # POST /doi/async/start wants an AsynchronousRequestBody (a DoiRequest), with
    # the Doi nested under "doi" -- not the bare Doi object.
    request = {"concreteType": "org.sagebionetworks.repo.model.doi.v2.DoiRequest",
               "doi": payload}
    token = syn.restPOST("/doi/async/start", body=json.dumps(request))["token"]
    for _ in range(120):
        time.sleep(2)
        resp = syn.restGET(f"/doi/async/get/{token}")
        state = resp.get("jobState")
        if state in ("PROCESSING", "IN_PROGRESS"):
            continue
        if state == "FAILED":
            raise SystemExit(f"DOI job FAILED: {resp.get('errorMessage') or resp}")
        print("DOI minted:")
        print(json.dumps(resp, indent=2))
        return resp
    raise SystemExit("Timed out waiting for the DOI job to complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", help="entity synId (the file view) [prepare mode]")
    ap.add_argument("--to-version", type=int,
                    help="already-created snapshot version to mint the new DOI for [prepare mode]")
    ap.add_argument("--from-version", type=int, default=None,
                    help="version whose existing DOI to clone metadata from (default: current)")
    ap.add_argument("--names-from-doi", default=None,
                    help="DataCite DOI (e.g. 10.7303/syn74341293.1) to source correctly-encoded "
                         "creator names from -- replaces Synapse-mangled '?' names by position. "
                         "Synapse GET mangles non-ASCII, so this is how you get clean authors.")
    ap.add_argument("--out", default="prepared_doi.json",
                    help="where to write the prepared payload for review")
    ap.add_argument("--submit", action="store_true",
                    help="MINT the DOI from --payload (the reviewed JSON). Off by default.")
    ap.add_argument("--payload", help="reviewed JSON file to submit (required with --submit)")
    args = ap.parse_args()

    syn = login()

    if args.submit:
        if not args.payload:
            raise SystemExit("--submit requires --payload <reviewed JSON file>.")
        payload = json.load(open(args.payload))
        print(f"About to mint a DOI for {payload.get('objectId')} "
              f"version {payload.get('objectVersion')} with {len(payload.get('creators', []))} creators.")
        submit(syn, payload)
        return

    # prepare mode
    if not args.view or args.to_version is None:
        raise SystemExit("prepare mode needs --view and --to-version.")
    versions = entity_versions(syn, args.view)
    if versions and args.to_version not in versions:
        raise SystemExit(f"version {args.to_version} does not exist for {args.view} "
                         f"(available: {sorted(versions)}). Create the snapshot first.")
    existing = fetch_existing_doi(syn, args.view, args.from_version)
    payload = build_new_payload(existing, args.to_version)
    if args.names_from_doi:
        fixed = apply_datacite_names(payload, fetch_datacite_creators(args.names_from_doi))
        print(f"\nCorrected {len(fixed)} creator name(s) from DataCite {args.names_from_doi}:")
        for old, new in fixed:
            print(f"   {old!r}  ->  {new!r}")
        remaining = [c["creatorName"] for c in payload.get("creators", []) if "?" in c["creatorName"]]
        if remaining:
            print(f"   WARNING: {len(remaining)} name(s) still contain '?': {remaining}")
    summarize(payload, existing)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote prepared payload to {args.out}")
    print(f"Review it, then mint with:\n    python {os.path.basename(__file__)} --submit --payload {args.out}")


if __name__ == "__main__":
    main()
