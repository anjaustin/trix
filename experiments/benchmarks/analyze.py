#!/usr/bin/env python3
"""
Live Analysis for TriX Training

Generates insights, visualizations, and comparisons during training.
Run alongside training for real-time science.

Usage:
    python experiments/analyze.py --live       # Continuous analysis
    python experiments/analyze.py --report     # One-time report
    python experiments/analyze.py --compare    # Compare all models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

import numpy as np


def load_metrics(log_dir: Path) -> Dict:
    """Load all metrics from a training run."""
    metrics_file = log_dir / "metrics.jsonl"
    
    if not metrics_file.exists():
        return {"steps": [], "evals": [], "checkpoints": []}
    
    steps = []
    evals = []
    checkpoints = []
    
    with open(metrics_file) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('type') == 'eval':
                    evals.append(data)
                elif data.get('type') == 'checkpoint':
                    checkpoints.append(data)
                elif 'loss' in data:
                    steps.append(data)
            except:
                continue
    
    return {"steps": steps, "evals": evals, "checkpoints": checkpoints}


def analyze_training_dynamics(metrics: Dict) -> Dict:
    """Analyze training dynamics from metrics."""
    steps = metrics['steps']
    evals = metrics['evals']
    
    if not steps:
        return {"status": "no_data"}
    
    analysis = {}
    
    # Loss curve analysis
    losses = [s['loss'] for s in steps if 'loss' in s]
    if losses:
        analysis['loss'] = {
            'initial': losses[0],
            'final': losses[-1],
            'min': min(losses),
            'improvement': (losses[0] - losses[-1]) / losses[0] * 100,
            'trend': 'decreasing' if losses[-1] < losses[0] else 'increasing',
        }
        
        # Detect loss spikes
        loss_diff = np.diff(losses)
        spikes = np.where(loss_diff > np.std(loss_diff) * 3)[0]
        analysis['loss']['num_spikes'] = len(spikes)
    
    # Throughput analysis
    throughputs = [s['throughput'] for s in steps if 'throughput' in s]
    if throughputs:
        analysis['throughput'] = {
            'mean': np.mean(throughputs),
            'std': np.std(throughputs),
            'max': max(throughputs),
            'min': min(throughputs),
        }
    
    # Routing analysis
    entropies = [s['routing_entropy'] for s in steps if 'routing_entropy' in s]
    if entropies:
        analysis['routing'] = {
            'initial_entropy': entropies[0],
            'final_entropy': entropies[-1],
            'mean_entropy': np.mean(entropies),
            'entropy_trend': 'increasing' if entropies[-1] > entropies[0] else 'decreasing',
        }
    
    # Validation analysis
    if evals:
        val_losses = [e['val_loss'] for e in evals if 'val_loss' in e]
        val_ppls = [e['val_ppl'] for e in evals if 'val_ppl' in e]
        
        if val_losses:
            analysis['validation'] = {
                'best_loss': min(val_losses),
                'best_ppl': min(val_ppls) if val_ppls else None,
                'final_loss': val_losses[-1],
                'final_ppl': val_ppls[-1] if val_ppls else None,
                'num_evals': len(val_losses),
            }
    
    # Learning rate analysis
    lrs = [s['lr'] for s in steps if 'lr' in s]
    if lrs:
        analysis['learning_rate'] = {
            'max': max(lrs),
            'min': min(lrs),
            'final': lrs[-1],
        }
    
    # Gradient analysis
    grad_norms = [s['grad_norm'] for s in steps if 'grad_norm' in s]
    if grad_norms:
        analysis['gradients'] = {
            'mean_norm': np.mean(grad_norms),
            'max_norm': max(grad_norms),
            'num_clipped': sum(1 for g in grad_norms if g > 0.99),  # Assuming clip=1.0
        }
    
    return analysis


def generate_text_plot(values: List[float], width: int = 50, height: int = 10) -> str:
    """Generate ASCII plot of values."""
    if not values or len(values) < 2:
        return "  [Insufficient data for plot]"
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
    
    # Normalize to height
    normalized = [(v - min_val) / range_val * (height - 1) for v in values]
    
    # Subsample if too many points
    if len(normalized) > width:
        indices = np.linspace(0, len(normalized) - 1, width).astype(int)
        normalized = [normalized[i] for i in indices]
    
    lines = []
    for row in range(height - 1, -1, -1):
        line = "  │"
        for val in normalized:
            if int(val) == row:
                line += "█"
            elif int(val) > row:
                line += "│"
            else:
                line += " "
        lines.append(line)
    
    lines.append("  └" + "─" * len(normalized))
    lines.append(f"   {min_val:.4f}" + " " * (len(normalized) - 20) + f"{max_val:.4f}")
    
    return "\n".join(lines)


def compare_experiments(experiments: List[str]) -> str:
    """Compare multiple experiments."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT COMPARISON")
    lines.append("=" * 70)
    lines.append("")
    
    results = []
    
    for exp_name in experiments:
        log_dir = Path(f"experiments/logs/{exp_name}")
        metrics = load_metrics(log_dir)
        analysis = analyze_training_dynamics(metrics)
        
        if analysis.get('status') == 'no_data':
            continue
        
        results.append({
            'name': exp_name,
            'analysis': analysis,
        })
    
    if not results:
        return "No data available for comparison."
    
    # Compare losses
    lines.append("### Loss Comparison ###")
    lines.append("")
    lines.append(f"{'Model':<15} {'Initial':<12} {'Final':<12} {'Best':<12} {'Improvement':<12}")
    lines.append("-" * 65)
    
    for r in results:
        loss = r['analysis'].get('loss', {})
        lines.append(
            f"{r['name']:<15} "
            f"{loss.get('initial', 0):<12.4f} "
            f"{loss.get('final', 0):<12.4f} "
            f"{loss.get('min', 0):<12.4f} "
            f"{loss.get('improvement', 0):<12.1f}%"
        )
    
    lines.append("")
    
    # Compare validation
    lines.append("### Validation Comparison ###")
    lines.append("")
    lines.append(f"{'Model':<15} {'Best PPL':<12} {'Final PPL':<12}")
    lines.append("-" * 40)
    
    for r in results:
        val = r['analysis'].get('validation', {})
        best_ppl = val.get('best_ppl', 'N/A')
        final_ppl = val.get('final_ppl', 'N/A')
        
        if isinstance(best_ppl, float):
            best_ppl = f"{best_ppl:.2f}"
        if isinstance(final_ppl, float):
            final_ppl = f"{final_ppl:.2f}"
        
        lines.append(f"{r['name']:<15} {best_ppl:<12} {final_ppl:<12}")
    
    lines.append("")
    
    # Compare throughput
    lines.append("### Throughput Comparison ###")
    lines.append("")
    lines.append(f"{'Model':<15} {'Mean':<15} {'Max':<15}")
    lines.append("-" * 45)
    
    for r in results:
        tp = r['analysis'].get('throughput', {})
        lines.append(
            f"{r['name']:<15} "
            f"{tp.get('mean', 0):>12,.0f} tok/s "
            f"{tp.get('max', 0):>12,.0f} tok/s"
        )
    
    lines.append("")
    
    # Compare routing
    lines.append("### Routing Entropy Comparison ###")
    lines.append("")
    lines.append(f"{'Model':<15} {'Initial':<12} {'Final':<12} {'Trend':<12}")
    lines.append("-" * 50)
    
    for r in results:
        routing = r['analysis'].get('routing', {})
        lines.append(
            f"{r['name']:<15} "
            f"{routing.get('initial_entropy', 0):<12.4f} "
            f"{routing.get('final_entropy', 0):<12.4f} "
            f"{routing.get('entropy_trend', 'N/A'):<12}"
        )
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_experiment_report(exp_name: str) -> str:
    """Generate detailed report for single experiment."""
    log_dir = Path(f"experiments/logs/{exp_name}")
    metrics = load_metrics(log_dir)
    analysis = analyze_training_dynamics(metrics)
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"EXPERIMENT REPORT: {exp_name.upper()}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")
    
    if analysis.get('status') == 'no_data':
        lines.append("No training data available yet.")
        return "\n".join(lines)
    
    # Loss section
    if 'loss' in analysis:
        loss = analysis['loss']
        lines.append("### Training Loss ###")
        lines.append(f"  Initial: {loss['initial']:.4f}")
        lines.append(f"  Final:   {loss['final']:.4f}")
        lines.append(f"  Best:    {loss['min']:.4f}")
        lines.append(f"  Improvement: {loss['improvement']:.1f}%")
        lines.append(f"  Spikes detected: {loss['num_spikes']}")
        lines.append("")
        
        # Loss curve
        losses = [s['loss'] for s in metrics['steps'] if 'loss' in s]
        if losses:
            lines.append("  Loss Curve:")
            lines.append(generate_text_plot(losses))
        lines.append("")
    
    # Validation section
    if 'validation' in analysis:
        val = analysis['validation']
        lines.append("### Validation ###")
        lines.append(f"  Best Loss: {val['best_loss']:.4f}")
        lines.append(f"  Best PPL:  {val['best_ppl']:.2f}" if val['best_ppl'] else "  Best PPL:  N/A")
        lines.append(f"  Final Loss: {val['final_loss']:.4f}")
        lines.append(f"  Final PPL:  {val['final_ppl']:.2f}" if val['final_ppl'] else "  Final PPL:  N/A")
        lines.append("")
    
    # Throughput section
    if 'throughput' in analysis:
        tp = analysis['throughput']
        lines.append("### Throughput ###")
        lines.append(f"  Mean: {tp['mean']:,.0f} tok/s")
        lines.append(f"  Max:  {tp['max']:,.0f} tok/s")
        lines.append(f"  Std:  {tp['std']:,.0f} tok/s")
        lines.append("")
    
    # Routing section
    if 'routing' in analysis:
        routing = analysis['routing']
        lines.append("### Routing Dynamics ###")
        lines.append(f"  Initial Entropy: {routing['initial_entropy']:.4f}")
        lines.append(f"  Final Entropy:   {routing['final_entropy']:.4f}")
        lines.append(f"  Mean Entropy:    {routing['mean_entropy']:.4f}")
        lines.append(f"  Trend: {routing['entropy_trend']}")
        lines.append("")
        
        # Entropy curve
        entropies = [s['routing_entropy'] for s in metrics['steps'] if 'routing_entropy' in s]
        if entropies:
            lines.append("  Routing Entropy Curve:")
            lines.append(generate_text_plot(entropies))
        lines.append("")
    
    # Gradient section
    if 'gradients' in analysis:
        grad = analysis['gradients']
        lines.append("### Gradients ###")
        lines.append(f"  Mean Norm: {grad['mean_norm']:.4f}")
        lines.append(f"  Max Norm:  {grad['max_norm']:.4f}")
        lines.append(f"  Clipped:   {grad['num_clipped']} steps")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def live_analysis(interval: int = 60):
    """Run continuous live analysis."""
    print("Starting live analysis...")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop")
    print()
    
    experiments = ['tiny', 'small', 'medium', 'code']
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        # Generate comparison report
        report = compare_experiments(experiments)
        
        # Save and display
        with open(results_dir / "live_comparison.txt", 'w') as f:
            f.write(report)
        
        print("\033[2J\033[H")  # Clear screen
        print(report)
        
        # Generate individual reports
        for exp in experiments:
            exp_report = generate_experiment_report(exp)
            with open(results_dir / f"report_{exp}.txt", 'w') as f:
                f.write(exp_report)
        
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Analyze TriX training")
    parser.add_argument("--live", action="store_true", help="Continuous live analysis")
    parser.add_argument("--report", type=str, help="Generate report for experiment")
    parser.add_argument("--compare", action="store_true", help="Compare all experiments")
    parser.add_argument("--interval", type=int, default=60, help="Update interval for live mode")
    
    args = parser.parse_args()
    
    if args.live:
        try:
            live_analysis(args.interval)
        except KeyboardInterrupt:
            print("\nAnalysis stopped.")
    elif args.report:
        print(generate_experiment_report(args.report))
    elif args.compare:
        print(compare_experiments(['tiny', 'small', 'medium', 'code']))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
