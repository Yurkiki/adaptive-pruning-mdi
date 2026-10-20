# Adaptive Pruning and Out-of-Sample MDI

Does a variable importance measured on **adaptively pruned** trees and
evaluated on **held-out data** beat the standard MDI?

Project for E. Scornet. The problem, the aim and the measures are in
[SPEC.md](SPEC.md).

## Setup

Python 3.10 or later.

```sh
pip install -r requirements.txt
python -m pytest tests/ -q
```
