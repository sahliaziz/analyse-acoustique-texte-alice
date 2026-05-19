#! /bin/bash

# conda install
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# bash ~/miniconda.sh -b -p $HOME/miniconda
# source ~/miniconda/bin/activate
# conda init
source $(conda info --base)/etc/profile.d/conda.sh
conda init bash

# conda env install
conda create -n py2k -y python=2 scipy numpy matplotlib
conda activate py2k
conda create -n py3k -y python=3 pip numpy
conda activate py3k
conda install pip
pip install django
pip install wheel
conda install pandas
python3 -m pip install webrtcvad
conda install -c conda-forge -y librosa
pip install praat-parselmouth --upgrade-strategy only-if-needed
# git installation (necessary for reaper)
sudo apt install git-all
# cmake installation (necessary for reaper)
sudo apt install cmake
# reaper installation
rm -rf 'REAPER'
git clone https://github.com/google/REAPER.git
cd REAPER
mkdir build
cd build
cmake ..
make
pwd
export PATH=$PATH:$(pwd)
echo "export PATH=$PATH:$(pwd)" >>~/.bashrc
# MUST THEN BE ADDED TO PATH ENVIRONMENT IF PREVIOUS LINE FAILED
