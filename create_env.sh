echo "Creating p38 conda env"
conda create -n p38 python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate p38

echo "Installing project requirements"
pip install -r ./requirements.txt

echo "Installing torch/conda binaries"
conda install pytorch torchvision -c pytorch -y

echo "Testing torch installation"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

echo "Installing low-level audio libraries"
apt-get update -y
apt install ffmpeg
apt-get install alsa-utils -y
apt-get install espeak libespeak1 -y