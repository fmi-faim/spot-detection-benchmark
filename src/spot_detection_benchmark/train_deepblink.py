import argparse

parser = argparse.ArgumentParser(description="Train deepBlink model")
parser.add_argument("--config", 
                    type=str,
                    help="Path to the configuration YAML file")
from deepblink.training import run_experiment

args = parser.parse_args()

import yaml
from deepblink.training import run_experiment

def get_values(dct):
    """Flatten deepBlink's {description, value} config to just the values."""
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else get_values(v)
        for k, v in dct.items()
    }

with open(args.config) as f:
    cfg = yaml.safe_load(f)

cfg = get_values(cfg)
run_experiment(cfg)


