# CO Decoder Experiments

This repository contains reproducible experiments for comparing author-provided decoders and proposed decoders for graph-based combinatorial optimization problems.

The first experiment focuses on the Maximum Clique problem using EGN, HGS, and GCON. The initial implementation starts with an EGN-style unsupervised maximum-clique model and the EGN author-style conditional-expectation decoder.

## Datasets

Initial Maximum Clique datasets:

- RB-small
- RB-large
- IMDB-BINARY
- COLLAB
- TWITTER ego networks

## Current status

- Phase 1: EGN author-decoder baseline.
- Phase 2: HGS author-decoder baseline.
- Phase 3: GCON author-decoder baseline.
- Phase 4: Proposed decoder comparison.
