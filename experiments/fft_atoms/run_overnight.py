#!/usr/bin/env python3
"""
FFT Atoms: Overnight Runner
===========================

Runs all atom tests and produces a summary report.

CODENAME: ANN WILSON - HEART Battery

Usage:
    python experiments/fft_atoms/run_overnight.py
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import subprocess
import json
from datetime import datetime
from pathlib import Path
import time


def run_atom(name, script_path):
    """Run a single atom test and capture output."""
    print(f"\n{'=' * 70}")
    print(f"RUNNING: {name}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )
    
    elapsed = time.time() - start_time
    
    return {
        'name': name,
        'script': script_path,
        'return_code': result.returncode,
        'elapsed_seconds': elapsed,
    }


def main():
    """Run all atom tests."""
    
    print("=" * 70)
    print("FFT ATOMS: OVERNIGHT BATTERY")
    print("CODENAME: ANN WILSON")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    atoms_dir = Path('/workspace/trix_latest/experiments/fft_atoms')
    results_dir = Path('/workspace/trix_latest/results/fft_atoms')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define atoms to run
    atoms = [
        ('ADDRESS', atoms_dir / 'atom_address.py'),
        ('BUTTERFLY_CORE', atoms_dir / 'atom_butterfly.py'),
    ]
    
    # Run each atom
    run_results = []
    
    for name, script in atoms:
        if script.exists():
            result = run_atom(name, str(script))
            run_results.append(result)
        else:
            print(f"WARNING: {script} not found, skipping {name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERNIGHT BATTERY SUMMARY")
    print("=" * 70)
    
    for r in run_results:
        status = "✓" if r['return_code'] == 0 else "✗"
        print(f"  {status} {r['name']}: {r['elapsed_seconds']:.1f}s")
    
    # Load detailed results from JSON files
    print("\n[DETAILED RESULTS]")
    
    json_files = list(results_dir.glob('atom_*.json'))
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Get most recent result for each atom
    seen_atoms = set()
    latest_results = {}
    
    for jf in json_files:
        atom_type = jf.stem.split('_')[1]  # e.g., 'address' from 'atom_address_20241216...'
        if atom_type not in seen_atoms:
            seen_atoms.add(atom_type)
            with open(jf) as f:
                latest_results[atom_type] = json.load(f)
    
    # Print summary table
    print("\n" + "-" * 50)
    print(f"{'Atom':<20} {'Accuracy':<15} {'Passed':<10}")
    print("-" * 50)
    
    all_passed = True
    
    for atom_type, data in latest_results.items():
        acc = data['aggregate']['mean_accuracy']
        passed = data['aggregate']['passed']
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{atom_type.upper():<20} {acc:.1%}{'':>10} {status}")
        if not passed:
            all_passed = False
    
    print("-" * 50)
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERNIGHT BATTERY: ✓ ALL ATOMS PASSED")
    else:
        print("OVERNIGHT BATTERY: ✗ SOME ATOMS FAILED")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'atoms_run': [r['name'] for r in run_results],
        'all_passed': all_passed,
        'results': {k: {
            'accuracy': v['aggregate']['mean_accuracy'],
            'passed': v['aggregate']['passed'],
        } for k, v in latest_results.items()},
    }
    
    summary_file = results_dir / f"overnight_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
