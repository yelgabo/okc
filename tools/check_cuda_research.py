"""Check the H200 digest's section coverage, local links and evidence identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def sections(text):
    headings = []
    fence = None
    lines = text.splitlines()
    for number, line in enumerate(lines, 1):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker[1][0]
            fence = None if fence == token else token if fence is None else fence
        elif fence is None and re.match(r'^#{1,6} ', line):
            headings.append({'heading': line, 'start': number})
    return [{**row, 'end': headings[i + 1]['start'] - 1 if i + 1 < len(headings) else len(lines)}
            for i, row in enumerate(headings)]


def digest(path, normalize_lf=False):
    data = path.read_bytes()
    if normalize_lf:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def prose_units(text):
    units = []
    for block in re.split(r'\n\s*\n', text.strip()):
        units.extend(block.splitlines() if block.startswith('|') else [block])
    return [hashlib.sha256(unit.encode()).hexdigest() for unit in units]


def validate(root, catalog_path):
    catalog = json.loads(catalog_path.read_text())
    folder = catalog_path.parent
    errors = []
    source = folder / catalog['source']['path']
    if digest(source) != catalog['source']['sha256']:
        errors.append('Original attachment changed; reconcile every topic')
    actual = sections(source.read_text())
    recorded = [{key: row[key] for key in ('heading', 'start', 'end')}
                for row in catalog['sections']]
    if actual != recorded:
        errors.append('Source section coverage differs from the catalog')
    notes = set(catalog['notes'])
    for row in catalog['sections']:
        if row['note'] not in notes:
            errors.append('Unmapped section: ' + row['heading'])
    for note in sorted(notes):
        path = folder / note
        if not path.is_file():
            errors.append('Missing note: ' + note)
            continue
        claim = catalog.get('claims', {}).get(note)
        if claim and prose_units(path.read_text()) != claim['units']:
            errors.append('Note changed: ' + note + '; reconcile claims and dependencies')
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = urlsplit(match[1])
            if target.scheme or not target.path:
                continue
            if not (path.parent / unquote(target.path)).exists():
                errors.append('Broken local link in ' + note + ': ' + target.path)
    for pin in catalog['evidence']:
        path = root / pin['path']
        if not path.is_file() or digest(path, pin.get('normalize_lf', False)) != pin['sha256']:
            errors.append('Evidence changed: ' + pin['path'] + '; review ' + ', '.join(pin['notes']))
    return errors, catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors, catalog = validate(args.root, args.root / 'docs/research/cuda-h200/catalog.json')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f"PASS: {len(catalog['sections'])} source sections, {len(catalog['notes'])} notes, "
          f"{len(catalog['evidence'])} evidence pins; local links resolve")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
