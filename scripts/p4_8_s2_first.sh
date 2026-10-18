#!/bin/bash
# P4-8 (D176(d), manager: "S2 goes first"): keep the preempted MB1 / MB2 jobs HELD until S2 is running, so S2 takes the
# first free L40S; then release them. Usage: setsid nohup bash scripts/p4_8_s2_first.sh S2_JOB MB_JOB... &
S2=$1; shift
until squeue -j "$S2" -h -o %T 2>/dev/null | grep -q RUNNING; do
  squeue -j "$S2" -h 2>/dev/null | grep -q . || break                 # S2 gone (finished / failed): stop holding
  sleep 60
done
scontrol release "$@" && echo "$(date +%m-%d_%H:%M) S2 ($S2) running -> released $*"
