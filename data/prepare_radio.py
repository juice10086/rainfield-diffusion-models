"""Build a reproducible z01 pairing/split manifest; leave raw data untouched."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.radio_dataset import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/home/czh/PPData5D-success')
    parser.add_argument('--output', default='data/ppdata5d_z01.json')
    parser.add_argument('--source-frequency', default='f00')
    parser.add_argument('--target-frequency', default='f03')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    args = parser.parse_args()
    manifest = build_manifest(args.data, args.source_frequency, args.target_frequency,
                              args.seed, args.val_fraction, args.test_fraction)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), counts=manifest['counts'],
                          splits=dict(Counter(r['split'] for r in manifest['records']))), indent=2))


if __name__ == '__main__':
    main()
