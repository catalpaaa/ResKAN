mamba env create -f environment.yml
mamba activate test
mamba env config vars set CUDA_HOME=$CONDA_PREFIX
mamba activate test
pip install lion-pytorch