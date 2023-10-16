# motif_discovery
This repository implements the algorithm for discovering repeated patterns in symbolic music as described in the paper: Yo-Wei Hsiao, Tzu-Yun Hung, Tsung-Ping Chen, Li Su, "BPS-Motif: A Dataset for Repeated Pattern Discovery of Polyphonic Symbolic Music," International Society of Music Information Retrieval Conference (ISMIR), November 2023. 

# Requirements
- numpy
- pretty_midi
- [Beethoven Piano Sonata Motif Dataset](https://github.com/Wiilly07/Beethoven_motif)
- [Baseline algorithms](https://github.com/wsgan001/repeated_pattern_discovery)

# Usage
1. Clone all required datasets and repositories, and put all data in the same file.
2. Execute *experiments.py*.
3. For the algorithm described in the paper, see *find_motives* in *SIA.py*.
