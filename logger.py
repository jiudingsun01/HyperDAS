import logging
import os

import torch
import torch.distributed as dist
from dotenv import load_dotenv

load_dotenv(override=True)


class DistributedAwareLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.log_on_all_ranks = os.environ.get("LOG_ALL_RANKS", "False") == "True"

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if dist.is_initialized():
            if dist.get_rank() == 0 or self.log_on_all_ranks:
                super()._log(level, msg, args, exc_info, extra, stack_info)
        else:
            super()._log(level, msg, args, exc_info, extra, stack_info)


logging.setLoggerClass(DistributedAwareLogger)
logging.basicConfig(level=logging.INFO)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

    return logger


if os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG":
    torch.set_printoptions(profile="full")
