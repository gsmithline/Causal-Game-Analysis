#!/bin/bash
# Run aspiration vs openai_5.2_none in 20 batches of 100 games each.
# Tries both directions. If openai_5.2_none as P1 hits the content filter,
# skips it and continues with the next batch.

for i in $(seq 1 20); do
    echo "=== Batch $i/20: openai_5.2_none (P1) vs aspiration (P2) ==="
    python evaluation/run_evaluation.py --pair openai_5.2_none aspiration --openai-models 5.2:none -n 100 -t 1 || echo "  (skipped - content filter)"

    echo "=== Batch $i/20: aspiration (P1) vs openai_5.2_none (P2) ==="
    python evaluation/run_evaluation.py --pair aspiration openai_5.2_none --openai-models 5.2:none -n 100 -t 1
done

echo "Done! 20 batches complete."
