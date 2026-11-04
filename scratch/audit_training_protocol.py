#!/usr/bin/env python3
"""Parse training result CSVs and produce a concise protocol summary.
Outputs:
- audit/training_protocol_summary.csv : one row per arm with columns:
    arm,final_epoch,final_mAP50,peak_mAP50,learning_rate_stage2,lr0_stage2,seed_present
- audit/two_stage_control_check.txt : indicates whether a baseline two‑stage control exists for Arm 3.
"""
import csv, re
from pathlib import Path

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def summarize_arm(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        return None
    # Determine final epoch (last row) and peak mAP50 (max of metrics/mAP50)
    final_row = rows[-1]
    final_epoch = final_row.get('epoch') or final_row.get('Epoch') or str(len(rows))
    final_mAP = final_row.get('metrics/mAP50(B)') or final_row.get('metrics/mAP50') or ''
    # Peak mAP across all rows
    mAP_vals = []
    for r in rows:
        val = r.get('metrics/mAP50(B)') or r.get('metrics/mAP50')
        if val:
            try:
                mAP_vals.append(float(val))
            except ValueError:
                pass
    peak_mAP = max(mAP_vals) if mAP_vals else ''
    # Learning rate columns – look for lr/pg2 (stage2) or lr/pg0 if only one.
    lr_stage2 = ''
    lr0_stage2 = ''
    # Assume the CSV contains columns like lr/pg0, lr/pg1, lr/pg2 per epoch.
    # Grab the last epoch values for each.
    if rows:
        last = rows[-1]
        lr_stage2 = last.get('lr/pg2') or last.get('lr/pg1') or last.get('lr/pg0') or ''
        # For lr0 (initial LR for stage2) we can inspect the first epoch where pg2 appears.
        for r in rows:
            if 'lr/pg2' in r:
                lr0_stage2 = r['lr/pg2']
                break
    # Seed presence – many logs do not contain a column, so we check for a column named 'seed' or a comment.
    seed_present = 'seed' in rows[0] if rows else False
    return {
        'final_epoch': final_epoch,
        'final_mAP50': final_mAP,
        'peak_mAP50': peak_mAP,
        'lr_stage2': lr_stage2,
        'lr0_stage2': lr0_stage2,
        'seed_present': seed_present,
    }

def main():
    results_dir = Path('results/tables')
    out_csv = Path('audit/training_protocol_summary.csv')
    fieldnames = ['arm','final_epoch','final_mAP50','peak_mAP50','lr_stage2','lr0_stage2','seed_present']
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for csv_file in results_dir.glob('arm*_*.csv'):
            arm_match = re.search(r'arm\d+', csv_file.name)
            arm = arm_match.group(0) if arm_match else csv_file.stem
            summary = summarize_arm(csv_file)
            if summary:
                row = {'arm': arm, **summary}
                writer.writerow(row)
    # Two‑stage control check for Arm 3
    # Look for a CSV that represents a baseline two‑stage (same config without SS2D).
    baseline_two_stage = any('baseline' in p.name.lower() for p in results_dir.glob('arm3*_*.csv'))
    ctrl_path = Path('audit/two_stage_control_check.txt')
    with ctrl_path.open('w') as f:
        if baseline_two_stage:
            f.write('Baseline two‑stage control for Arm 3 FOUND.\n')
        else:
            f.write('Baseline two‑stage control for Arm 3 NOT FOUND (Issue 4).\n')
    print('Training protocol summary and two‑stage control check written.')

if __name__ == '__main__':
    main()

