#!/usr/bin/env python3
"""
Parallel Training Launcher for TriX Experiments

Launches all 4 models in parallel on Thor (128GB allows this).
Monitors progress and generates live reports.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --dry-run  # Show what would run
    python experiments/run_all.py --monitor  # Just monitor existing runs
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import queue


@dataclass
class ExperimentRun:
    """Track a running experiment."""
    name: str
    config_path: str
    data_dir: str
    log_dir: str
    checkpoint_dir: str
    process: Optional[subprocess.Popen] = None
    start_time: Optional[datetime] = None
    status: str = "pending"


EXPERIMENTS = [
    ExperimentRun(
        name="trix-tiny",
        config_path="experiments/configs/tiny.yaml",
        data_dir="experiments/data/fineweb_5m",
        log_dir="experiments/logs/tiny",
        checkpoint_dir="experiments/checkpoints/tiny",
    ),
    ExperimentRun(
        name="trix-small",
        config_path="experiments/configs/small.yaml",
        data_dir="experiments/data/fineweb_50m",
        log_dir="experiments/logs/small",
        checkpoint_dir="experiments/checkpoints/small",
    ),
    ExperimentRun(
        name="trix-medium",
        config_path="experiments/configs/medium.yaml",
        data_dir="experiments/data/fineweb_500m",
        log_dir="experiments/logs/medium",
        checkpoint_dir="experiments/checkpoints/medium",
    ),
    ExperimentRun(
        name="trix-code",
        config_path="experiments/configs/code.yaml",
        data_dir="experiments/data/stack_250m",
        log_dir="experiments/logs/code",
        checkpoint_dir="experiments/checkpoints/code",
    ),
]


class ExperimentMonitor:
    """
    Real-time monitoring of parallel experiments.
    
    Generates periodic reports and alerts.
    """
    
    def __init__(self, experiments: List[ExperimentRun]):
        self.experiments = experiments
        self.report_dir = Path("experiments/results")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.running = True
    
    def get_latest_metrics(self, exp: ExperimentRun) -> Dict:
        """Read latest metrics from experiment log."""
        metrics_file = Path(exp.log_dir) / "metrics.jsonl"
        
        if not metrics_file.exists():
            return {"status": "no_data"}
        
        # Read last few lines
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return {"status": "empty"}
            
            # Get latest step metric and latest eval metric
            latest_step = None
            latest_eval = None
            
            for line in reversed(lines):
                try:
                    data = json.loads(line.strip())
                    if data.get('type') == 'eval' and latest_eval is None:
                        latest_eval = data
                    elif 'loss' in data and latest_step is None:
                        latest_step = data
                    
                    if latest_step and latest_eval:
                        break
                except:
                    continue
            
            return {
                "status": "running",
                "latest_step": latest_step,
                "latest_eval": latest_eval,
                "total_lines": len(lines),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_report(self) -> str:
        """Generate a status report for all experiments."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"TRIX TRAINING STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        
        for exp in self.experiments:
            lines.append(f"### {exp.name.upper()} ###")
            lines.append(f"Status: {exp.status}")
            
            if exp.start_time:
                elapsed = (datetime.now() - exp.start_time).total_seconds()
                lines.append(f"Elapsed: {elapsed/3600:.2f} hours")
            
            metrics = self.get_latest_metrics(exp)
            
            if metrics.get("latest_step"):
                step_data = metrics["latest_step"]
                lines.append(f"Step: {step_data.get('step', '?')}")
                lines.append(f"Loss: {step_data.get('loss', '?'):.4f}")
                lines.append(f"Throughput: {step_data.get('throughput', '?'):,.0f} tok/s")
                lines.append(f"Tokens: {step_data.get('tokens', '?'):,}")
            
            if metrics.get("latest_eval"):
                eval_data = metrics["latest_eval"]
                lines.append(f"Val Loss: {eval_data.get('val_loss', '?'):.4f}")
                lines.append(f"Val PPL: {eval_data.get('val_ppl', '?'):.2f}")
            
            lines.append("")
        
        # GPU Memory
        try:
            import torch
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                lines.append(f"GPU Memory: {mem_used:.1f} / {mem_total:.1f} GB")
        except:
            pass
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_report(self):
        """Save report to file and print to console."""
        report = self.generate_report()
        
        # Save to file
        report_file = self.report_dir / "live_status.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Also save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.report_dir / f"status_{timestamp}.txt", 'w') as f:
            f.write(report)
        
        # Print to console
        print("\033[2J\033[H")  # Clear screen
        print(report)
    
    def check_for_issues(self) -> List[str]:
        """Check for training issues that need attention."""
        issues = []
        
        for exp in self.experiments:
            metrics = self.get_latest_metrics(exp)
            
            if metrics.get("latest_step"):
                step_data = metrics["latest_step"]
                
                # Check for NaN
                loss = step_data.get('loss', 0)
                if loss != loss:  # NaN check
                    issues.append(f"{exp.name}: NaN loss detected!")
                
                # Check for explosion
                if loss > 100:
                    issues.append(f"{exp.name}: Loss explosion ({loss:.2f})")
                
                # Check for zero throughput
                throughput = step_data.get('throughput', 0)
                if throughput < 100:
                    issues.append(f"{exp.name}: Low throughput ({throughput:.0f} tok/s)")
        
        return issues
    
    def monitor_loop(self, interval: int = 60):
        """Run monitoring loop."""
        print("Starting monitoring loop...")
        print(f"Reports saved to: {self.report_dir}")
        print(f"Update interval: {interval} seconds")
        print("Press Ctrl+C to stop")
        print()
        
        while self.running:
            self.save_report()
            
            issues = self.check_for_issues()
            if issues:
                print("\n⚠️  ISSUES DETECTED:")
                for issue in issues:
                    print(f"  - {issue}")
            
            # Check if all experiments are done
            all_done = all(
                exp.status in ["completed", "failed", "pending"]
                for exp in self.experiments
            )
            
            if all_done and any(exp.status == "completed" for exp in self.experiments):
                print("\n✅ All experiments completed!")
                self.generate_final_report()
                break
            
            time.sleep(interval)
    
    def generate_final_report(self):
        """Generate final summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("TRIX TRAINING - FINAL REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)
        lines.append("")
        
        results = []
        
        for exp in self.experiments:
            summary_file = Path(exp.log_dir) / "summary.json"
            
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                
                final = summary.get('final_metrics', {})
                results.append({
                    'name': exp.name,
                    'val_ppl': final.get('final_val_ppl', '?'),
                    'best_ppl': final.get('best_val_ppl', '?'),
                    'params': final.get('model_params', '?'),
                    'tokens': final.get('total_tokens', '?'),
                    'duration': summary.get('duration_seconds', 0) / 3600,
                })
        
        # Results table
        lines.append("### RESULTS ###")
        lines.append("")
        lines.append(f"{'Model':<15} {'Params':<12} {'Tokens':<12} {'Best PPL':<10} {'Time (hrs)':<10}")
        lines.append("-" * 60)
        
        for r in results:
            params = f"{r['params']/1e6:.1f}M" if isinstance(r['params'], (int, float)) else str(r['params'])
            tokens = f"{r['tokens']/1e6:.0f}M" if isinstance(r['tokens'], (int, float)) else str(r['tokens'])
            ppl = f"{r['best_ppl']:.2f}" if isinstance(r['best_ppl'], (int, float)) else str(r['best_ppl'])
            time_hrs = f"{r['duration']:.2f}" if isinstance(r['duration'], (int, float)) else str(r['duration'])
            
            lines.append(f"{r['name']:<15} {params:<12} {tokens:<12} {ppl:<10} {time_hrs:<10}")
        
        lines.append("")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        # Save
        with open(self.report_dir / "FINAL_REPORT.txt", 'w') as f:
            f.write(report)
        
        # Also save as JSON
        with open(self.report_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(report)


def launch_experiment(exp: ExperimentRun) -> subprocess.Popen:
    """Launch a single experiment as subprocess."""
    cmd = [
        sys.executable,
        "experiments/train.py",
        "--config", exp.config_path,
        "--data-dir", exp.data_dir,
        "--log-dir", exp.log_dir,
        "--checkpoint-dir", exp.checkpoint_dir,
    ]
    
    # Create log directory
    Path(exp.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Open log file for stdout/stderr
    log_file = open(Path(exp.log_dir) / "stdout.log", 'w')
    
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
    )
    
    return process


def run_all(dry_run: bool = False, monitor_only: bool = False):
    """Launch all experiments in parallel."""
    
    if monitor_only:
        monitor = ExperimentMonitor(EXPERIMENTS)
        try:
            monitor.monitor_loop(interval=30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        return
    
    print("=" * 70)
    print("TRIX PARALLEL TRAINING LAUNCHER")
    print("=" * 70)
    print()
    
    # Check data exists
    missing_data = []
    for exp in EXPERIMENTS:
        data_path = Path(exp.data_dir) / "train.bin"
        if not data_path.exists():
            missing_data.append(exp.name)
    
    if missing_data:
        print("⚠️  Missing data for experiments:")
        for name in missing_data:
            print(f"  - {name}")
        print()
        print("Run data preparation first:")
        print("  python experiments/prepare_data.py --all")
        print()
        
        if not dry_run:
            response = input("Prepare data now? [y/N] ")
            if response.lower() == 'y':
                subprocess.run([sys.executable, "experiments/prepare_data.py", "--all"])
            else:
                return
    
    if dry_run:
        print("DRY RUN - Would launch:")
        for exp in EXPERIMENTS:
            print(f"  - {exp.name}: {exp.config_path}")
        return
    
    # Launch all experiments
    print("Launching experiments...")
    processes = []
    
    for exp in EXPERIMENTS:
        print(f"  Starting {exp.name}...")
        exp.process = launch_experiment(exp)
        exp.start_time = datetime.now()
        exp.status = "running"
        processes.append(exp.process)
        time.sleep(2)  # Stagger launches slightly
    
    print()
    print(f"All {len(EXPERIMENTS)} experiments launched!")
    print()
    
    # Start monitoring
    monitor = ExperimentMonitor(EXPERIMENTS)
    
    def check_processes():
        """Background thread to check process status."""
        while True:
            all_done = True
            for exp in EXPERIMENTS:
                if exp.process:
                    ret = exp.process.poll()
                    if ret is None:
                        all_done = False
                    elif ret == 0:
                        exp.status = "completed"
                    else:
                        exp.status = "failed"
            
            if all_done:
                monitor.running = False
                break
            
            time.sleep(10)
    
    # Start background checker
    checker = threading.Thread(target=check_processes, daemon=True)
    checker.start()
    
    # Run monitor
    try:
        monitor.monitor_loop(interval=60)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Stopping experiments...")
        for exp in EXPERIMENTS:
            if exp.process:
                exp.process.terminate()
        print("Experiments terminated.")


def main():
    parser = argparse.ArgumentParser(description="Launch parallel TriX training")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--monitor", action="store_true", help="Monitor existing runs")
    
    args = parser.parse_args()
    
    run_all(dry_run=args.dry_run, monitor_only=args.monitor)


if __name__ == "__main__":
    main()
