# -*- coding: utf-8 -*-
"""Run inference of a LeRobot policy inside the LIBERO simulation environment.

This script mirrors the usage pattern from :mod:`lerobot.record` for invoking a
pretrained policy, while reusing utilities from ``run_libero_eval`` for the
simulation set‑up.  A pretrained LeRobot policy is loaded and evaluated on a
LIBERO benchmark task suite.  For each step the policy receives observations
from the simulator and outputs an action that is sent back to the environment.

Example
-------
```bash
python -m lerobot.libero.lerobot_inference \
    --policy_path=lerobot/some_policy \
    --task_suite_name=libero_spatial \
    --num_trials_per_task=1
```
"""

from __future__ import annotations
import os

os.environ["MUJOCO_GL"] = "glfw"

import dataclasses
import logging
from typing import Tuple

import numpy as np
import draccus
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from enum import Enum
import os
from pathlib import Path
import torch
import random
import time 
import imageio

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DATE = time.strftime("%Y_%m_%d")
os.environ["MUJOCO_GL"] = "glfw"

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

INFERENCE_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 10,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 7,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 100,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 7,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 100,  # longest training demo has 373 steps
}

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis–angle format."""
    # Implementation copied from ``libero_utils.quat2axisangle`` to avoid an
    # extra dependency on the ``experiments`` package.
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def _get_libero_env(task, seed, resolution: int) -> Tuple[OffScreenRenderEnv, str]:
    """Initialise LIBERO environment and return it with the task description."""
    task_description = task.language
    base_bddl = Path(get_libero_path("bddl_files"))
    task_bddl = base_bddl / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def _get_images(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract third‑person and wrist camera images from observations."""
    full = obs["agentview_image"][::-1, ::-1]
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    return full, wrist


def _dummy_action() -> list:
    """Return a no‑op action used while waiting for the scene to stabilise."""
    return [0, 0, 0, 0, 0, 0, -1]


def _prepare_observation(obs: dict, image_size: int) -> Tuple[dict, np.ndarray]:
    """Format simulator observations into the structure expected by the policy."""
    full, wrist = _get_images(obs)
    # full_resized = np.array(Image.fromarray(full).resize((image_size, image_size)))
    full_resized = (
        np.array(Image.fromarray(full).resize((image_size, image_size)))
        .astype(np.float32)
    )
    # wrist_resized = np.array(Image.fromarray(wrist).resize((image_size, image_size)))
    wrist_resized = (
        np.array(Image.fromarray(wrist).resize((image_size, image_size)))
        .astype(np.float32)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    # print("state:", state, len(state))
    observation = {
        "observation.images.image": full_resized,
        "observation.images.wrist_image":    wrist_resized,
        "observation.state":                 state,
    }
    return observation, full


def _load_policy(path: str):
    """Load a pretrained LeRobot policy from ``path``."""
    policy_cfg = PreTrainedConfig.from_pretrained(path)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(path, config=policy_cfg)
    return policy, policy_cfg


def _run_episode(cfg: LiberoInferenceConfig, env, task_desc: str, policy, policy_cfg, image_size: int) -> bool:
    """Run a single episode and return whether it succeeded."""
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    obs = env.reset()
    t = 0
    replay_images = []
    # wait for objects to stabilise
    while t < cfg.num_steps_wait:
        obs, _, _, _ = env.step(_dummy_action())
        t += 1
    success = False
    policy.reset()
    while t < max_steps:
        observation, img = _prepare_observation(obs, image_size)
        replay_images.append(img)
        action_tensor = predict_action(
            observation,
            policy,
            get_safe_torch_device(policy_cfg.device),
            policy_cfg.use_amp,
            task=task_desc,
            act_len=cfg.act_len
        )
        action = action_tensor.cpu().numpy()
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    
    return success, replay_images

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclasses.dataclass
class LiberoInferenceConfig:
    policy_path: str
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    env_img_res: int = 256
    act_len: int = 100

    seed: int = 7

def save_rollout_video(seed, rollout_images, act_len, idx, success, task_description, task_suite_name, log_file=None):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = f"./rollouts/{DATE_TIME}_act_{act_len}_{task_suite_name}/{processed_task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    # log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

# @draccus.wrap()
def run_libero_inference() -> float:
    """Evaluate a LeRobot policy on a LIBERO task suite."""
    from draccus.argparsing import parse
    cfg = parse(config_class=LiberoInferenceConfig)
    log_dir = f"./rollouts/{DATE_TIME}_act_{cfg.act_len}_{cfg.task_suite_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inference.log")

    # Configure root logger to output to both console and file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any default handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Common formatter for both handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    logging.basicConfig(level=logging.INFO)
    policy, policy_cfg = _load_policy(cfg.policy_path)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    total_episodes = 0
    total_successes = 0
    for task_id in range(task_suite.n_tasks):
        task_episodes = 0
        task_successes = 0
        task = task_suite.get_task(task_id)
        task_description = task.language
        env, task_desc = _get_libero_env(task, cfg.seed, cfg.env_img_res)
        for _ in range(cfg.num_trials_per_task):
            success, replay_images = _run_episode(cfg, env, task_desc, policy, policy_cfg, cfg.env_img_res)
            total_successes += int(success)
            task_successes += int(success)
            total_episodes += 1
            task_episodes += 1
            task_success_rate = task_successes / max(task_episodes, 1)

            # save_rollout_video(
            #         seed, replay_images, total_episodes, success=success, task_description=task_description, task_suite_name=task_suite_name                    
            #     )
            save_rollout_video(
                cfg.seed,
                replay_images,
                act_len=cfg.act_len,
                idx=total_episodes,
                success=success,
                task_description=task_description,
                task_suite_name=cfg.task_suite_name
            )
        logging.info(f"{task_description} success rate: {task_success_rate}")
    success_rate = total_successes / max(total_episodes, 1)
    logging.info("Overall success rate: %.4f", success_rate)
    return success_rate


if __name__ == "__main__":

    run_libero_inference()
