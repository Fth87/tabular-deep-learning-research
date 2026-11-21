# Autoresearch for RankUp augmentation

This folder adapts the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) cycle: an agent forms a hypothesis, edits `candidate.py`, tests the candidate on a fixed protocol, records the score, and then decides keep/discard. This system is **not** a clone of upstream, which does LLM optimization of nanochat, and it is not an agent that forms hypotheses on its own after the Codex session ends. The file `program.md` describes the limits and rules.

`candidate.py` is the candidate under test; `snapshots/round*.py` keeps the versions from earlier rounds. `runner.py` uses the same RankUp loop as the baseline, `modal_ar.py` sends jobs to Modal, and `score.py` checks the nine conditions and the input checksums before producing a decision. Results are in `../hasil/autoresearch/` and `../hasil/autoresearch-round*/`. No checkpoints were saved in this pilot.

Commands from the `4-augmentasi-input` folder, after a candidate has been chosen and tested:

```bash
uv run modal run --profile lagi usulan/autoresearch/modal_ar.py --action check
uv run modal run --detach --profile lagi usulan/autoresearch/modal_ar.py --action run --run-id autoresearch-round-N --parallel 2
uv run modal volume get experiment-ta-results autoresearch-round-N usulan/hasil --profile lagi --force
uv run python usulan/autoresearch/score.py autoresearch-round-N
```

Use a new run ID for every change so that different results do not get mixed. The second command queues 9 jobs; wait for all jobs to finish before fetching results. `--detach` keeps the Modal jobs alive, not an agent process that would find new candidates by itself. Do not run many rounds without controlling cost. The old validation set has been reused repeatedly and the historical test set is already exposed; a single-seed keep/discard decision is only screening and needs a split/holdout and extra seeds before any research claim.
