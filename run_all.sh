#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -rf "$SCRIPT_DIR/simulator_results"

QPS_VALUES=(12.0 9.0 6.0 4.0 3.0 1.5)
SCH_VALUES=("lor" "round_robin" "llq")

for qps in "${QPS_VALUES[@]}"; do
    for sch in "${SCH_VALUES[@]}"; do
      if [[ $# -eq 1 ]]; then
        bash "$SCRIPT_DIR/run_single.sh" $sch $qps $1
      else
        bash "$SCRIPT_DIR/run_single.sh" $sch $qps "retrace"
      fi
    done
done

python3 plot.py --path "$SCRIPT_DIR/simulator_results" --test_names sharegpt_lor --test_names sharegpt_round_robin --test_names sharegpt_llq