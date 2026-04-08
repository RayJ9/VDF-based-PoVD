from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PACKAGE_DIR.parent


def _load_submodule(qualified_name: str, source_filename: str):
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing

    source_path = _REPO_DIR / source_filename
    spec = importlib.util.spec_from_file_location(qualified_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {qualified_name} from {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_load_submodule("povd_platform.settings", "settings.py")
_load_submodule("povd_platform.crypto", "crypto.py")
_core = _load_submodule("povd_platform.core", "core.py")
_load_submodule("povd_platform.cli", "cli.py")


BaseMiner = _core.BaseMiner
BaselineVDFMiner = _core.BaselineVDFMiner
Block = _core.Block
MinerStepReport = _core.MinerStepReport
MiningPlatform = _core.MiningPlatform
MiningPlatformConfig = _core.MiningPlatformConfig
MiningPlatformResult = _core.MiningPlatformResult
MiningTick = _core.MiningTick
Network = _core.Network
NetworkEvent = _core.NetworkEvent
Platform = _core.Platform
PlatformConfig = _core.PlatformConfig
PlatformResult = _core.PlatformResult
PoVDMiner = _core.PoVDMiner
PoWMiner = _core.PoWMiner
build_platform_config = _core.build_platform_config
build_genesis_block = _core.build_genesis_block
compare_mining_modes = _core.compare_mining_modes
compare_platforms = _core.compare_platforms
format_comparison = _core.format_comparison
format_mining_result = _core.format_mining_result
format_platform_result = _core.format_platform_result
run_mining_platform = _core.run_mining_platform
run_platform = _core.run_platform


__all__ = [
    "BaseMiner",
    "BaselineVDFMiner",
    "Block",
    "MinerStepReport",
    "MiningPlatform",
    "MiningPlatformConfig",
    "MiningPlatformResult",
    "MiningTick",
    "Network",
    "NetworkEvent",
    "Platform",
    "PlatformConfig",
    "PlatformResult",
    "PoVDMiner",
    "PoWMiner",
    "build_platform_config",
    "build_genesis_block",
    "compare_mining_modes",
    "compare_platforms",
    "format_comparison",
    "format_mining_result",
    "format_platform_result",
    "run_mining_platform",
    "run_platform",
]
