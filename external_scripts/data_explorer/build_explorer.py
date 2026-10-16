#!/usr/bin/env python3
"""Render profile.json (from analyze_phenotype.py) into the Data Explorer page.

Usage:
  python3 build_explorer.py PROFILE.json OUT.html --label "Bridge2AI-Voice Pediatric v2.0.0" [--force]

The page title and header show --label, and the build paths in the payload (meta.root,
meta.sites_from, meta.org_csv) are replaced or dropped. See README.md in this folder.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('profile'); ap.add_argument('out')
ap.add_argument('--label', required=True)
ap.add_argument('--template', default=os.path.join(HERE, 'explorer_template.html'))
ap.add_argument('--force', action='store_true')
a = ap.parse_args()
if os.path.exists(a.out) and not a.force:
    sys.exit('refusing to overwrite existing %s (pass --force)' % a.out)
data = json.load(open(a.profile, encoding='utf-8'))
data['meta']['root'] = a.label
for k in ('sites_from', 'org_csv', 'disposition_counts'):
    data['meta'].pop(k, None)
payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
html = open(a.template, encoding='utf-8').read()
for ph in ('__DATA__', '__TITLE__', '__ROOT__'):
    if html.count(ph) != 1:
        sys.exit('template must contain %s exactly once' % ph)
esc = a.label.replace('&', '&amp;').replace('<', '&lt;')
html = html.replace('__TITLE__', esc).replace('__ROOT__', esc).replace('const DATA = __DATA__;', 'const DATA = ' + payload + ';')
open(a.out, 'w', encoding='utf-8').write(html)
print('wrote %s (%.1f MB)' % (a.out, os.path.getsize(a.out) / 1e6), file=sys.stderr)
