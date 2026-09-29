"""讀取 markets.yaml — 所有參數的單一事實來源。"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "markets.yaml"


def load_raw() -> dict:
    """整份 markets.yaml（給 taiwan_exclude 這類不屬於單一市場的設定用）。"""
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def load(market: str = "taiwan") -> dict:
    cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    if market not in cfg:
        raise KeyError(f"markets.yaml 沒有 '{market}'，可用：{[k for k in cfg if k != 'meta']}")
    return cfg[market]


def params(market: str = "taiwan") -> dict:
    return load(market)["params"]


def costs(market: str = "taiwan") -> dict:
    return load(market)["costs"]


def profiles(market: str = "taiwan") -> dict:
    """出場設定檔（中線／長線）。沒定義就回空 dict。"""
    return load(market).get("profiles", {})


if __name__ == "__main__":
    import json
    for m in ("crypto", "taiwan"):
        print(f"=== {m} ===")
        print(json.dumps(params(m), ensure_ascii=False, indent=2))
