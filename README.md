# LeRobot-LIBERO

**Run and evaluate Hugging Face’s LeRobot policies on the LIBERO benchmark for lifelong robot learning.**

LeRobot-LIBERO bridges [Hugging Face's LeRobot](https://github.com/huggingface/lerobot) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), enabling evaluation of vision-language-actions (VLAs) models and imitation learning policies on standardized robotic manipulation tasks. This repository supports policy inference, reproducible evaluation, and integration with LIBERO's task suites.

---

![License](https://img.shields.io/github/license/JiahongChen/lerobot-libero)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:JiahongChen/lerobot-libero.git
cd lerobot-libero
```

### 2. Create and Activate Conda Environment

```bash
conda create -y -n lerobot-libero python=3.10
conda activate lerobot-libero
```

### 3. Install LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda install -y ffmpeg -c conda-forge
pip install -e .
pip install -e ".[smolvla, pi0]"
cd ..
```

### 4. Install LIBERO

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

### 5. Install Additional Dependencies

```bash
pip install -r libero_requirements.txt
```

---

## 🚀 Running Inference

Use the following command to run inference on a selected LIBERO task suite:

```bash
python lerobot_inference.py \
    --policy_path peeeeeter/smolvla_spatial \
    --task_suite_name [libero_spatial | libero_object | libero_goal | libero_10 | libero_90]
```

Replace `--policy_path` with your desired pretrained policy and select an appropriate `--task_suite_name`.

---

## 📁 Repository Structure

- `lerobot_inference.py`: Entry point for evaluating LeRobot models on LIBERO tasks.
- `libero_requirements.txt`: Dependencies for running LIBERO tasks with LeRobot.
- Other files coming soon...

---

## 📌 Notes

- Ensure you are using Python 3.10 for compatibility with both LeRobot and LIBERO.
- `ffmpeg` is required for video-related operations and is installed via conda.
- The script supports evaluating pretrained models hosted on Hugging Face Hub or locally.

---

## 🤝 Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot): Hugging Face's large behavior model framework.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): A benchmark suite for lifelong robot learning.

