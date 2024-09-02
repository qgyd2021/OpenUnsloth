## OpenUnsloth

### 参考文档

```text

https://github.com/unslothai/unsloth

```

### 创建容器

```text

docker run -itd \
--name open_unsloth \
--network host \
--gpus all \
python:3.11-slim /bin/bash



docker build -t open_unsloth:v20240902_1055 .

docker run -itd \
--name open_unsloth \
--network host \
open_unsloth:v20240902_1055



查看GPU
nvidia-smi
watch -n 1 -d nvidia-smi

```


### 创建环境

```text
conda create --name open_unsloth \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate open_unsloth

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

```
