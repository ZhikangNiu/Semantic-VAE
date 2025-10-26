<h1 align="center"><strong>Semantic-VAE: Semantic-Alignment Latent Representation for Better Speech Synthesis</strong></h1>

<p align="center" style="font-size: 1 em; margin-top: 1em">
<a href="https://zhikangniu.github.io/">Zhikang Niu<sup>1,2</sup></a>,
<a href="">Shujie Hu<sup>3</sup></a>,
<a href="">Jeongsoo Choi<sup>4<sup></a>,
<a href="">Yushen Chen<sup>1,2<sup></a>,
<a href="">Peining Chen<sup>1<sup></a>,
<a href="">Pengcheng Zhu<sup>5<sup></a>,
<a href="">Yunting Yang<sup>5<sup></a>,
<a href="">Bowen Zhang<sup>5<sup></a>,
<a href="">Jian Zhao<sup>5<sup></a>,
<a href="">Chunhui Wang<sup>5<sup></a>,
<a href="https://chenxie95.github.io/">Xie Chen<sup>1,2<sup></a>
</p>

<p align="center">
  <sup>1</sup>MoE Key Lab of Artificial Intelligence, X-LANCE Lab, School of Computer Science,
  <br> Shanghai Jiao Tong University, China
  <sup>2</sup>Shanghai Innovation Institute, China <br>
  <sup>3</sup>The Chinese University of Hong Kong, China <br>
  <sup>4</sup>Korea Advanced Institute of Science and Technology, South Korea
  <sup>5</sup>Geely, China &nbsp;&nbsp; <br>
</p>

<div align="center">
  <a href="https://github.com/ZhikangNiu/Semantic-VAE">
    <img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="Python">
  </a>
  <a href="https://arxiv.org/abs/2509.22167">
    <img src="https://img.shields.io/badge/arXiv-2509.22167-b31b1b.svg?logo=arXiv" alt="arXiv">
  </a>
  <a href="https://huggingface.co/zkniu/Semantic-VAE/tree/main/semantic_vae">
    <img src="https://img.shields.io/badge/Model-semantic--vae-FF6B6B.svg?logo=HuggingFace" alt="semantic-vae">
  </a>
  <a href="https://huggingface.co/zkniu/Semantic-VAE/tree/main/acoustic_vae_dim16">
    <img src="https://img.shields.io/badge/Model-acoustic--vae--dim16-6BA4FF.svg?logo=HuggingFace" alt="acoustic-vae-dim16">
  </a>
  <a href="https://huggingface.co/zkniu/Semantic-VAE/tree/main/acoustic_vae_dim64">
    <img src="https://img.shields.io/badge/Model-acoustic--vae--dim64-4ECDC4.svg?logo=HuggingFace" alt="acoustic-vae-dim64">
  </a>
</div>

## 📜 News
🚀 [2025.10] We upload Semantic-VAE 1M update steps model in [Huggingface](https://huggingface.co/zkniu/Semantic-VAE/tree/main/semantic_vae_1000k).

🚀 [2025.10] We release all the code and pre-trained models(semantic-vae, acoustic-vae-dim16, and acoustic-vae-dim64) to promote the research of semantic-aligned VAE for speech synthesis after cleaning the code.

## 💡 Highlights
1. **Semantic Alignment**: A novel VAE framework that utilizes semantic alignment regularization to mitigate the reconstruction-generation optimization dilemma in high-dimensional latent spaces.
2. **Plug and Play for VAE-based TTS**: Semantic-VAE can be easily integrated into existing VAE-based TTS models, providing a simple yet effective way to improve their performance.
3. **Accelerated Training**: Semantic-VAE significantly accelerates the convergence of VAE-based TTS models while maintaining the same inference speed as the original models.
4. **High-Quality Speech Generation**: Semantic-VAE achieves high-quality speech generation with improved intelligibility and speaker similarity, making it suitable for various TTS applications.

## 🛠️ Usage
### 1. Install environment and dependencies
```bash
# We recommend using conda to create a new environment.
conda create -n semantic-vae python=3.11
conda activate semantic-vae

git clone https://github.com/ZhikangNiu/Semantic-VAE.git
cd Semantic-VAE

# Install PyTorch >= 2.2.0, e.g.,
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install audiotools: https://github.com/descriptinc/audiotools
pip install git+https://github.com/descriptinc/audiotools

# Install editable version of Semantic-VAE
pip install -e .
```

### 2. Feature Extraction
We use pre-trained SSL models (WavLM) to extract features for semantic alignment. You can use the following command to extract features after preparing the dataset.
```bash
bash extract_ssl_features.sh
```

### 3. Train the model
If you want to train your custom VAE, simply modify the `conf/svae/example.yml` file as a template and specify the path to your customized .yml file. And our training command is as follow (you can also check `train.sh`):
```shell
torchrun --nproc_per_node 8 scripts/train.py --args.load custom_yml_path
```

### 4. Run Inference

You can download the model and follow the example in [`examples/example.py`](examples/example.py) to reconstruct audio using the pretrained model.

For multi-GPU processing, we also provide two utility scripts:

- [`extract_latent.py`](scripts/extract_latent.py): Extracts latent representations from audio files.
- [`recon_latent_to_wave.py`](scripts/recon_latent_to_wave.py): Reconstructs waveforms from latent codes.


## ❤️ Acknowledgments
Our work is built upon the following open-source projects [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec), [F5-TTS](https://github.com/SWivid/F5-TTS) and [BigVGAN](https://github.com/NVIDIA/BigVGAN/tree/main). Thanks to the authors for their great work, and if you have any questions, you can first check them on their respective issues.

## ✒️ Citation and License
Our code is released under MIT License. If our work and codebase is useful for you, please cite as:
```
@article{niu2025semantic,
  title={Semantic-VAE: Semantic-Alignment Latent Representation for Better Speech Synthesis},
  author={Niu, Zhikang and Hu, Shujie and Choi, Jeongsoo and Chen, Yushen and Chen, Peining and Zhu, Pengcheng and Yang, Yunting and Zhang, Bowen and Zhao, Jian and Wang, Chunhui and others},
  journal={arXiv preprint arXiv:2509.22167},
  year={2025}
}
