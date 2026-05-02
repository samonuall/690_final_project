# LLM Reward Design Experiments

This project studies how LLM-designed reward functions perform in gridworld tasks, with an emphasis on alignment and misspecification.

## Obfuscated environment inputs

We provide non-runnable, code-like .txt summaries of the environments to give LLMs structural context without exposing reward logic or alignment cues.

- Location: [src/envs/obfuscated](src/envs/obfuscated)
- Files: [src/envs/obfuscated/boat_race_obfuscated.txt](src/envs/obfuscated/boat_race_obfuscated.txt), [src/envs/obfuscated/distributional_shift_obfuscated.txt](src/envs/obfuscated/distributional_shift_obfuscated.txt)
- Contents: layout, symbols, and observation encoding details only (no reward/performance hints)
