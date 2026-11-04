#!/usr/bin/env python3
"""Verify that Arm 3 Stage 2 uses the reported learning rate 0.0005.
Outputs:
- audit/lr_stage2_check.txt : PASS/FAIL with observed lr0.
"""
import csv
from pathlib import Path

def main():
    # Locate the Arm 3 two‑stage CSV
    csv_path = Path('results/tables/arm3_case_b_two_stage_results.csv')
    if not csv_path.is_file():
        print('Arm3 CSV not found')
        return
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Find the first epoch where lr/pg2 appears (stage2 learning rate)
    lr0 = None
    for r in rows:
        if 'lr/pg2' in r:
            try:
                lr0 = float(r['lr/pg2'])
            except ValueError:
                lr0 = None
            break
    out_path = Path('audit/lr_stage2_check.txt')
    with out_path.open('w') as f:
        if lr0 is None:
            f.write('FAIL: lr/pg2 column not found or value unparsable.\n')
        elif abs(lr0 - 0.0005) < 1e-6:
            f.write(f'PASS: Stage2 learning rate=0.0005 (found {lr0}).\n')
        else:
            f.write(f'FAIL: Stage2 learning rate differs (found {lr0}).\n')
    print('LR stage2 check written.')

if __name__ == '__main__':
    main()

