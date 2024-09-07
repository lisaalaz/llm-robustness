import logging
import pprint

from tap import Tap

from helpers.utils import LOGGER_NAME, load_config, setup

logger = logging.getLogger(LOGGER_NAME)


class Arguments(Tap):
    config: str
    log_level: str
    seed: int


if __name__ == "__main__":
    args = Arguments(explicit_bool=True).parse_args()
    setup(args.log_level, args.seed)
    config = load_config(args.config)
    logger.info(f"\nInput config:\n{pprint.pformat(config, sort_dicts=False)}\n")

    from helpers.pipelines import run_pipeline

    run_pipeline(config)
