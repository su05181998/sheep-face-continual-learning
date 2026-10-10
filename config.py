from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = deepcopy(config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent)
    return config


def resolve_path(config: Dict[str, Any], value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config["_project_root"]) / path).resolve()
