#!/bin/bash
# Run aspiration vs openai_5.4_medium — 3 batches of 100 games each direction.
# Currently at 1800 (aspiration P1) and 1700 (openai P1), need to reach 2k.
# Directions run separately so one failing doesn't block the other.
# Both directions launch in parallel within each batch.

for i in $(seq 1 3); do
    echo "=== Batch $i/3 ==="

    # Launch both directions in parallel
    python evaluation/run_evaluation.py --pair aspiration openai_5.4_medium --openai-models 5.4:medium -n 100 -t 1 &
    PID1=$!
    python evaluation/run_evaluation.py --pair openai_5.4_medium aspiration --openai-models 5.4:medium -n 100 -t 1 &
    PID2=$!

    # Wait for both to finish (don't exit on failure)
    wait $PID1 || echo "  aspiration(P1) vs openai_5.4_medium(P2) failed in batch $i"
    wait $PID2 || echo "  openai_5.4_medium(P1) vs aspiration(P2) failed in batch $i"
done

echo "Done! 3 batches complete."
