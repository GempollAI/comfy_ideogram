import json
from typing import Optional


class Configs:
    def __init__(self):
        self.proxy_url: Optional[str] = None
        self.api_key: Optional[str] = None

    @staticmethod
    def from_json(file_path: str) -> "Configs":
        with open(file_path, "r") as f:
            config_dict = json.load(f)

        configs = Configs()
        configs.proxy_url = config_dict.get("proxy_url")
        configs.api_key = config_dict.get("api_key")
        return configs


CONFIGS = Configs.from_json("./configs.json")