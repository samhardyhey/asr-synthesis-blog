echo "Creating p38 conda env"
conda create -n p38 python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate p38

echo "Installing project requirements"
pip install -r ./requirements.txt

echo "Install torch/conda binaries"
conda remove pytorch torchvision -y
pip uninstall torch -y
pip uninstall torch -y  # yes twice
conda install pytorch torchvision -c pytorch -y

echo "Testing torch installation"
python -c 'import torch; torch.cuda.is_available(); torch.cuda.device_count()'