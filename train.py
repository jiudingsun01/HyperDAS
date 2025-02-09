import gc
import os
import random
from datetime import datetime

import hydra
import omegaconf
import torch
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from logger import get_logger
from src.hyperdas.data import get_axbench_collate_fn, get_ravel_collate_fn
from src.hyperdas.llama3.model import RavelInterpretorHypernetwork
from src.hyperdas.utils import NamedDataLoader

logger = get_logger(__name__)


def run_experiment(
    config: DictConfig,
    device: str | torch.DeviceObjType = "cuda",
):
    # If experiment config exists, use it instead of root config
    config = config.experiment if hasattr(config, "experiment") else config
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    if config.training.debug_model:
        config.model.inference_modes = ["groundtruth"]

    if "default" in config.model.inference_modes:
        config.model.inference_modes.remove("default")
        config.model.inference_modes.append(None)

    config.wandb_config.run_name = (
        config.wandb_config.run_name
        or f"{config.model.subspace_module}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if config.wandb_config.log:
        wandb.init(
            project=config.wandb_config.project,
            entity=config.wandb_config.entity,
            name=config.wandb_config.run_name,
            config=OmegaConf.to_container(config),
            group=config.wandb_config.group,
            tags=config.wandb_config.tags,
            notes=config.wandb_config.notes,
        )
        wandb.run.log_code("/workspace/HyperDAS/src/hyperdas/")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(config.dataset.train_path)

    # Load multiple test sets
    if isinstance(config.dataset.test_path, (list, omegaconf.ListConfig)):
        test_set = [load_from_disk(path) for path in config.dataset.test_path]
    else:
        test_set = load_from_disk(config.dataset.test_path)

    if config.dataset.dataset_type == "axbench":
        collate_fn = get_axbench_collate_fn(tokenizer)
    elif config.dataset.dataset_type == "ravel":
        collate_fn = get_ravel_collate_fn(
            tokenizer,
            source_suffix_visibility=config.dataset.source_suffix_visibility,
            base_suffix_visibility=config.dataset.base_suffix_visibility,
            contain_entity_position=True,
            add_space_before_target=True,
        )

    # very hacky
    try:
        getattr(HydraConfig.get().job, "num")
        is_multirun = True
    except Exception:
        is_multirun = False
    num_workers = 0 if is_multirun else config.training.num_workers

    train_data_loader = DataLoader(
        train_set,
        batch_size=config.training.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if isinstance(test_set, list):
        test_data_loader = [
            NamedDataLoader(
                data_loader=DataLoader(
                    test_set_,
                    batch_size=config.training.test_batch_size,
                    collate_fn=collate_fn,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
                name=os.path.basename(config.dataset.test_path[i]),
            )
            for i, test_set_ in enumerate(test_set)
        ]
    else:
        test_data_loader = [
            NamedDataLoader(
                data_loader=DataLoader(
                    test_set,
                    batch_size=config.training.test_batch_size,
                    collate_fn=collate_fn,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
                name=os.path.basename(config.dataset.test_path),
            )
        ]

    hypernetwork = RavelInterpretorHypernetwork(config, device)

    if config.training.load_trained_from is not None:
        logger.info(f"Loading model from {config.training.load_trained_from}")
        hypernetwork.load_model(config.training.load_trained_from)

    hypernetwork.run_train(train_loader=train_data_loader, test_loader=test_data_loader)

    if config.wandb_config.log:
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    try:
        logger.info(f"Working directory : {os.getcwd()}")
        logger.info(f"Original working directory    : {get_original_cwd()}")
        logger.info(f"to_absolute_path('foo')       : {to_absolute_path('foo')}")
        logger.info(f"to_absolute_path('/foo')      : {to_absolute_path('/foo')}")

        # Check if we're running in serial mode
        is_serial = os.environ.get("LAUNCH_MODE", "parallel") == "serial"

        if cfg.device_mode is None:
            cfg.device_mode = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if cfg.device_mode == "mps":
            device = "mps"
        else:
            if is_serial:
                # Use single GPU for serial mode
                device = "cuda:0"
            elif cfg.device_mode == "cuda":
                # Use distributed GPUs for parallel mode
                num_gpus = torch.cuda.device_count()
                try:
                    job_num = getattr(HydraConfig.get().job, "num", 0)
                except Exception:
                    job_num = 0
                gpu_id = job_num % num_gpus
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"

        logger.info(
            f"Running in {'serial' if is_serial else 'parallel'} mode on device {device}"
        )

        if not is_serial and cfg.device_mode == "cuda":
            torch.cuda.set_device(device)

        # Set seed
        torch.manual_seed(cfg.training.seed)
        random.seed(cfg.training.seed)

        run_experiment(cfg, device)

        if cfg.device_mode == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"An error occurred in hydra_main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
