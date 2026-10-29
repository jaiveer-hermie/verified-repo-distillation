#!/bin/bash
# Kill stale eval/batch processes. Run as a script so the patterns cannot match
# this shell's own command line (that mistake killed the caller twice).
pkill -f 'batch1_nebius' 2>/dev/null
pkill -f 'run\.py eval' 2>/dev/null
sleep 2
echo "remaining eval processes: $(pgrep -fc 'run\.py eval' 2>/dev/null || echo 0)"
