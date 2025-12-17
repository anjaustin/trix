#!/bin/bash
#
# Mesa 9: Granville Test Launcher
# 
# Usage:
#   ./run_granville.sh        # Run in foreground
#   ./run_granville.sh &      # Run in background
#   nohup ./run_granville.sh > /dev/null 2>&1 &  # Run detached
#
# Monitor:
#   tail -f /workspace/trix_latest/results/granville/granville_*.log
#
# Check status:
#   ps aux | grep granville
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="/workspace/trix_latest/results/granville"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Set up environment
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "MESA 9: GRANVILLE TEST"
echo "============================================================"
echo "Start time: $(date)"
echo "Results dir: $RESULTS_DIR"
echo ""
echo "To monitor progress:"
echo "  tail -f $RESULTS_DIR/granville_*.log"
echo ""
echo "To stop:"
echo "  pkill -f granville_full_test.py"
echo "============================================================"
echo ""

# Run the test
cd "$SCRIPT_DIR"
python3 granville_full_test.py

echo ""
echo "============================================================"
echo "Test completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
