# PoVD Platform

PoVD Platform is a mining consensus platform for comparing PoVD, PoW, and VDF baseline miners under the same network and workload settings. It also includes mining power measurement and plotting utilities for analyzing fork behavior, thread activity, and power-related signals.

## Environment

Create an isolated environment and install the platform dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

You can also install the same dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Main Entry

Run a configured mining comparison:

```bash
python -m povd_platform compare --config-file your_config.json
```

Run a single mining mode:

```bash
python -m povd_platform run --mode povd --config-file your_config.json
```

## Parameter Study

The `study.py` module is a parameter sweep helper. It rewrites selected settings, runs repeated mining comparisons, and prints how PoVD and PoW fork rates change as the expected block interval changes.
