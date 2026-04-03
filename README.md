# PoVD Platform

## Setup

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternative:

```powershell
python -m pip install -r requirements.txt
```

If you are using a venv located at `.\povd_platform\.venv\`, you still need to install the package from the repo root:

```powershell
.\povd_platform\.venv\Scripts\python.exe -m pip install -e .
```

## Configure

Use a JSON config file. You can start from:

- `.\povd_platform\mining_demo_config.json` (small demo)

## Run

Compare all 3 modes:

```powershell
python -m povd_platform compare --config-file .\povd_platform\mining_demo_config.json --quiet
```

Run a single mode:

```powershell
python -m povd_platform run --mode povd --config-file .\povd_platform\mining_demo_config.json --quiet --json
```

Modes: `povd`, `pow`, `vdf_baseline`

Run the provided example script:

```powershell
python .\povd_platform\run_mining_example.py
```
