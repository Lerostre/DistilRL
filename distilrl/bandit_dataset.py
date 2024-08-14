import gc
import os
from pathlib import Path

import hydra
import numpy as np
from constants import CONFIG_DIR, DATASET_DIR, MODEL_DIR
from datasets import Dataset, load_from_disk
from hydra.utils import get_class
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from SMPyBandits.Environment import Evaluator
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data import generate_arm_dist, window_stack, dump_pkl
from utils.misc import seed_everything


@hydra.main(config_path="configs", config_name="bandit", version_base="1.2")
def bandit_dataset(cfg: DictConfig) -> None:

    # define configs and paths
    logger.info("Parsing bandit_dataset config")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(cfg.seed)

    env_cfg, arm_cfg, policies_cfg = [
        OmegaConf.to_container(cfg[sub_cfg])
        for sub_cfg in ["env", "arm", "policies"]
    ]

    # load trainer configs
    arm_cfg["arm_type"] = get_class(arm_cfg["arm_type"])
    policies = list(policies_cfg.values())
    for policy in policies:
        policy["archtype"] = get_class(policy["archtype"])

    # load tokenizer and model
    odd_set, even_set, uniform_set = [
        generate_arm_dist(
            n_arms=cfg.n_arms, n_runs=cfg.n_runs, strategy=strategy
        )
        for strategy in ["odd", "even", "uniform"]
    ]
    distributions = dict(
        zip(["odd", "even", "uniform"], [odd_set, even_set, uniform_set])
    )
    dump_pkl(distributions, Path(DATASET_DIR, "distributions.pkl"))
    logger.info("Arm distributions generated")

    # prepare dataset
    for name, dataset in distributions.items():

        envs = [
            {"arm_type": arm, "params": params}
            for arm, params in zip(
                [arm_cfg["arm_type"] for _ in range(cfg.n_runs)], dataset
            )
        ]
        env_cfg["environment"] = envs
        env_cfg["policies"] = policies

        logger.info(f"Started evaluating {dataset}")
        evaluator = Evaluator(env_cfg)
        for idx, env in tqdm(enumerate(evaluator.envs), desc="Running envs"):
            evaluator.startOneEnv(idx, env)
            # enable_print()

        rewards = np.zeros((len(policies), env_cfg["horizon"]))
        for i in range(len(policies)):
            rewards[i] = evaluator.getRewards(i)
        actions = evaluator.allPulls[0].argmax(1)
        time_steps = np.repeat(
            np.arange(env_cfg["horizon"])[None, :],
            repeats=len(policies),
            axis=0,
        )

        dataset = {}
        for field, array in zip(
            ["rewards", "actions", "time_steps"], [rewards, actions, time_steps]
        ):
            dataset[field] = window_stack(
                array[:, : -cfg.dataset.seq_len], (1, cfg.dataset.seq_len)
            ).squeeze(1)
            dataset["future_" + field] = window_stack(
                array[:, cfg.dataset.seq_len :], (1, cfg.dataset.seq_len)
            )[:, :, 0].squeeze(1)
        dataset["distribution"] = ["odd"] * len(dataset["rewards"])
        del dataset["future_time_steps"]
        gc.collect()

        dataset = Dataset.from_dict(dataset)
        dataset.set_format(type="torch")
        if cfg.dataset.sample[name] > 0:
            dataset = dataset.select(
                np.random.choice(len(dataset), size=(cfg.dataset.sample[name]))
            )
        logger.info(f"Created dataset `{name}_set`")
        print(dataset)

        dataset_path = Path(DATASET_DIR, f"{name}_set")
        dataset.save_to_disk(dataset_path)
        logger.info(f"Saved model to `{dataset_path}`")

    logger.info("Bandit dataset fully created")


bandit_dataset()

# poetry run python distilrl/bandit_dataset.py
