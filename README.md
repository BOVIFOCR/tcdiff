# TCDiff

> **TCDiff: Triple Condition Diffusion Model with 3D Constraints for Stylizing Synthetic Faces** </br>
> Bernardo Biesseck, Pedro Vidal, Luiz Coelho, Roger Granada, David Menotti </br>
> In SIBGRAPI 2024 </br>

[Paper](assets/TCDiff_2024.pdf) &nbsp; &nbsp; [Arxiv](https://arxiv.org/abs/2409.03600)

<img src="assets/face_mixer.png" alt="image" width="400" height="auto">
We propose a Triple Condition Diffusion Model (TCDiff) to improve face style transfer from real to synthetic faces through 2D and 3D facial constraints, enhancing face identity consistency while keeping the necessary high intra-class variance for training face recognition models with synthetic data.

</br></br>
### 1. Main requirements
- Python==3.8
- CUDA==11.2
- numpy==1.24.2
- mxnet==1.9.1
- torch>=2.2.0
- torchvision==0.12.0
- pytorch-lightning==1.7.1
- opencv-python>=4.8.1.78

### 2. Create environment
```
CONDA_ENV=tcdiff
conda create -y --name $CONDA_ENV python=3.9
conda activate $CONDA_ENV

conda env config vars set CUDA_HOME="/usr/local/cuda-11.2"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib64"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set PATH="$CUDA_HOME:$CUDA_HOME/bin:$LD_LIBRARY_PATH:$PATH"; conda deactivate; conda activate $CONDA_ENV
```

### 3. Clone this repository and install requirements
```
git clone https://github.com/BOVIFOCR/tcdiff.git
cd tcdiff
./install.sh   # install dependencies and download needed pre-trained models
```

### 4. Train model
```
cd tcdiff
bash src/scripts/train_with_3DMM_consistency_constraints.sh
```
Model `.ckpt` will be saved at folder `experiments_WITH_3DMM_CONSISTENCY_CONSTRAINTS/tcdiff/checkpoints/`

### 5. Create new synthetic face dataset
For simplicity, we provide here the 10k synthetic identities generated and used by [DCFace](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_DCFace_Synthetic_Face_Generation_With_Dual_Condition_Diffusion_Model_CVPR_2023_paper.pdf)
- [10k synthetic identities](https://www.dropbox.com/scl/fi/4enrbhdpok4cchtzk468m/dcface_10000_synthetic_ids.zip?rlkey=gayi92laaychxp4a6ok8axjde&st=psa6rezs&dl=0)
