
# PyTorch Setup Guide on 4th Gen Intel Xeon Scalable Processor
This repository provides a semantic segmentation model created using open data called [Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset).

It also introduces the source code for fine tuning based on the models and techniques to speed up the inference of the models.

## Prerequisites
- Ubuntu 22.04 LTS (GNU/Linux 5.19.17-051917-generic x86_64)
- Docker (*Installation procedure described below)
## Installing
### Install Docker
```Bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG
```

### Download Docker image
```Bash
docker pull continuumio/anaconda3
```
### Launch Docker container
The container is starting as root user. Also, port 8888 should be bounded to the host OS and the container.
```Bash
sudo docker run -it -u 0 --privileged -p 8888:8888 continuumio/anaconda3 /bin/bash
```
Thereafter, the work will be done on the container.
### Install additional software
```Bash
apt-get update
apt-get install -y wget unzip git vim numactl libgl1-mesa-dev libjemalloc-dev libgl1-mesa-dev build-essential

export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=$LD_PRELOAD:/opt/conda/lib/libiomp5.so
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export OMP_NUM_THREADS=<Number of physical cores>

conda create -n pytorch python=3.8 -y
conda activate pytorch
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch 
pip install transformers diffusers accelerate yacs opencv-python pycocotools defusedxml cityscapesscripts
pip install -U numpy==1.19
```
### Clone this repo
```Bash
cd ~
git clone https://github.com/hiouchiy/Setup_PyTorch_on_SPR.git
```
### Run Resnet50 training in FP32 and BF16
```
# FP32 (without AMX)
numactl -m 0 -N 0 python train_rn50_fp32.py

# BF16 (with AMX)
numactl -m 0 -N 0 python train_rn50_bf16.py
```
### Run Mask R-CNN training in FP32 and BF16
```
cd ~
git clone https://github.com/IntelAI/models.git
cd models/quickstart/object_detection/pytorch/maskrcnn/training/cpu
mkdir /root/dataset
export DATASET_DIR=/root/dataset
bash download_dataset.sh        # Maybe takes long time.

cd ~/models
export MODEL_DIR=$(pwd)

mkdir /root/output
export OUTPUT_DIR=/root/output

# Run a FP32 trarining
cd ${MODEL_DIR}/quickstart/object_detection/pytorch/maskrcnn/training/cpu
bash training.sh fp32

# Run a BF16 trarining
cd ${MODEL_DIR}/quickstart/object_detection/pytorch/maskrcnn/training/cpu
bash training.sh bf16
```
### Run Stable Diffusion inference in FP32 and BF16
```
# Run a FP32 trarining
cd ~
wget https://gitlab.com/juliensimon/huggingface-demos/-/raw/main/optimum/stable_diffusion_intel/sd_blog_1.py
numactl -m 0 -N 0 python sd_blog_1.py

# Run a BF16 trarining
cd ~
wget https://gitlab.com/juliensimon/huggingface-demos/-/raw/main/optimum/stable_diffusion_intel/sd_blog_3.py
numactl -m 0 -N 0 python sd_blog_3.py
```
## License
This project is licensed under the MIT License.

## Reference
PyTorch Performance Tuning Guide for Intel CPU: 
https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/performance_tuning/tuning_guide.html

Resnet50 Script:
https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/examples.html

Mask R-CNN script: 
https://github.com/IntelAI/models/tree/master/quickstart/object_detection/pytorch/maskrcnn/training/cpu

Stable Diffusion script:
https://huggingface.co/blog/stable-diffusion-inference-intel