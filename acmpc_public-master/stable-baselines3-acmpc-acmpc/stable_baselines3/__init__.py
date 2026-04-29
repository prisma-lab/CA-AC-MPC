import os

from stable_baselines3.ppo import PPO
from stable_baselines3.common.utils import get_system_info

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = ["PPO", "get_system_info", "__version__"]
