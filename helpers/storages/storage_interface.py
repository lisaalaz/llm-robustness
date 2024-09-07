import json
import logging
import os

from helpers.utils import LOGGER_NAME, get_full_path

from .combination import Combination

logger = logging.getLogger(LOGGER_NAME)


class StorageInterface:
    def __init__(self, directory_path: str):
        self.directory_path = get_full_path(directory_path)

        if os.path.isfile(self.directory_path):
            raise ValueError(f"{self.directory_path} is a file, should be a directory")

        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path, exist_ok=True)
            logger.info(f"Created storage at {self.directory_path}")

        logger.debug(f"Storage: {self.directory_path}")

    def load(self, combination: Combination) -> list[dict[str, str | float]] | None:
        if not self.exists(combination):
            return None

        combination_path = self.combination_path(combination)

        with open(combination_path, "r") as file:
            return json.load(file)

    def save(self, combination: Combination, data) -> None:
        combination_path = self.combination_path(combination)

        if self.exists(combination):
            logger.warning(f"Overwriting data at {combination_path}")
        else:
            os.makedirs(os.path.dirname(combination_path), exist_ok=True)

        with open(combination_path, "w") as file:
            json.dump(data, file, indent=2)
            file.write("\n")

        logger.info(f"Saved object to {combination_path}\n")

    def exists(self, combination: Combination) -> bool:
        combination_path = self.combination_path(combination)

        if not os.path.isfile(combination_path):
            return False

        return True

    def combination_path(self, combination: Combination) -> str:
        return os.path.join(self.directory_path, combination.path_name())
