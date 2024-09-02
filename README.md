## OpenUnsloth

### 参考文档

```text

https://github.com/unslothai/unsloth

```

### 创建容器

```text

docker run -itd --gpus all python:3.12-slim /bin/bash

```


### 创建环境

```text
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

```
