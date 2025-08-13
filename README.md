# lerobot-libero
Evaluating LeRobot-based model on LIBERO environment

```
git clone git@github.com:JiahongChen/lerobot-libero.git
cd lerobot-libero
conda create -y -n lerobot-libero python=3.10
conda activate lerobot-libero
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda install ffmpeg -c conda-forge
pip install -e .
pip install -e ".[smolvla, pi0]"
cd ..
```

Install LIBERO
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

Install dependencies
```
pip install -r ./libero_requirements.txt
```

Run inference
```
python lerobot_inference.py --policy_path=peeeeeter/lite_object_500k --task_suite_name=[libero_object|libero_10|libero_spatial|libero_goal|libero_90]
```

