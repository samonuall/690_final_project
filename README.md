# LLM Reward Design Experiments

This project studies how LLM-designed reward functions perform in gridworld tasks, with an emphasis on alignment and misspecification.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Clone and install the gridworld dependency as an editable package:
   ```bash
   git clone https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
   uv pip install -e ./ai-safety-gridworlds
   ```
3. Install project dependencies:
   ```bash
   uv sync
   ```
4. Copy `.env.example` to `.env` and add your OpenRouter API key.

## Obfuscated environment inputs

We provide non-runnable, code-like .txt summaries of the environments to give LLMs structural context without exposing reward logic or alignment cues.

- Location: [src/context/obfuscated](src/context/obfuscated)
- Files: [src/context/obfuscated/boat_race_obfuscated.txt](src/context/obfuscated/boat_race_obfuscated.txt), [src/context/obfuscated/distributional_shift_obfuscated.txt](src/context/obfuscated/distributional_shift_obfuscated.txt)
- Contents: layout, symbols, and observation encoding details only (no reward/performance hints)
