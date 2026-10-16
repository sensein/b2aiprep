# Data Explorer (`data_explorer.html`)

A single HTML page, shipped at the top level of each published tree, that summarizes every phenotype field. For each field it shows the description, how often it is blank, the distribution of its values and how many participants have a value. It needs no server: open it in any web browser.

The page is made in two steps from a finished release tree (a bundle or a deidentified tree):

1. `analyze_phenotype.py` reads every `phenotype/*/*.tsv` and its `.json` data dictionary, plus the `sub-*/sub-*_sessions.tsv` files if present, and writes `profile.json`. Only the Python standard library is needed (3.6+).
2. `build_explorer.py` puts the profile into `explorer_template.html` and writes the page. `--label` sets the title, and the build paths are removed from the page.

## Release command (as used for b2aiprep 4.0.0)

```bash
python3 external_scripts/data_explorer/analyze_phenotype.py <tree> profile.json --list-ids-max 10
python3 external_scripts/data_explorer/build_explorer.py profile.json <tree>/data_explorer.html \
  --label "Bridge2AI-Voice Adult v4.0.0 (registered access)"
```

Run it once per published tree, with that tree's label, for example "Bridge2AI-Voice Pediatric v2.0.0 (controlled access)".

## Notes

- **Rebuild after any phenotype change.** The page holds counts, not the data, so it goes stale when a table changes.
- **`--list-ids-max N`** lists, for each value held by at most N rows, the participant and session of those rows; the page shows them on hover. On a deidentified tree these are its pseudonyms. Use `--list-ids-max 0` to leave them out.
- **Not for released pages:** `--sites` (real site names) and `--org` (field-map dispositions) are for internal QA. On an internal tree the listed IDs and values would link participants to their answers.
- **Counts:** both participants (distinct `participant_id`) and rows are counted. Percentages and bars are shares of rows.
