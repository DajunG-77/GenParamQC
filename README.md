# Hybrid Diffusion-Optimization for Quantum Synthesis with Continuous Native Gates

Code repository for:

> **Hybrid diffusion-optimization for quantum synthesis with continuous native gates**
> *Physica Scripta*, IOP Publishing.
> [https://iopscience.iop.org/article/10.1088/1402-4896/ae6a4a](https://iopscience.iop.org/article/10.1088/1402-4896/ae6a4a)

We extend the discrete-gate diffusion compiler of GenQC to **continuous, hardware-native parameterized gates** (e.g. `Rx`, `Ry`, `Rxx`) and combine the diffusion-sampled circuit *structure* with a downstream gradient-based **parameter optimization** stage. The result is shorter, higher-fidelity circuits than the diffusion model alone, while still being orders of magnitude faster than running an optimizer from scratch.

---

The repository includes:

- [genqc_param/](genqc_param/) — the main package, containing the model pipeline, training and inference notebooks, and the post-diffusion parameter optimizer (`fast_optim.py`).
- [bench_utils/](bench_utils/) — examples for reproducing some figures of the paper, together with the test set of 3-qubit target unitaries (`targets_param.npy`).

---

## Installation

### Option A — Conda (recommended)

The code is developed and tested with **Python 3.12** and **PyTorch with CUDA 12.4** on Linux. CPU-only and macOS also work for inference but training is slow.

```bash
# 1) create and activate a fresh environment
conda create -n genqc-param python=3.12 -y
conda activate genqc-param

# 2) install PyTorch
# Linux / Windows with NVIDIA GPU (CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# macOS or CPU-only:
# pip install torch

# 3) install the remaining dependencies
pip install -r requirements.txt

# 4) register the env as a Jupyter kernel (optional but handy for the notebooks)
python -m ipykernel install --user --name genqc-param --display-name "Python (genqc-param)"
```

### Option B — uv

The project also ships `pyproject.toml` files for [uv](https://github.com/astral-sh/uv):

```bash
cd genqc_param
uv sync
cd ../bench_utils
uv sync
```

---

## Usage

### Training

Open and run [genqc_param/genQC-param-train.ipynb](genqc_param/genQC-param-train.ipynb). The notebook
1. loads the pretrained discrete-gate diffusion compiler from `saves/qc_unet_config_Compilation_3_qubit/`,
2. constructs a fine-tuning dataset of random circuits over the continuous-parameter gate pool,
3. fine-tunes the U-Net and writes the checkpoint to `saves/qc_params-new/`.

A CUDA-capable GPU is strongly recommended.

### Inference

Open and run [genqc_param/inference_fast_optim.ipynb](genqc_param/inference_fast_optim.ipynb). The notebook
1. loads `saves/qc_params-new/`,
2. reads target unitaries from `../bench_utils/targets_param.npy`,
3. for each target, samples candidate circuits with the diffusion model and picks the best one by process infidelity,
4. runs the gradient-based post-optimization (`fast_optim.py`) and plots both circuits side by side.

Process infidelity is reported as

$$1 - F_{\mathrm{pro}}(U_{\mathrm{target}}, U_{\mathrm{circuit}}) = 1 - \frac{|\mathrm{Tr}(U_{\mathrm{target}}^\dagger U_{\mathrm{circuit}})|^2}{d^2}.$$



---

## Citation

If you find this code or the method useful, please cite:

```bibtex
@article{10.1088/1402-4896/ae6a4a,
  title   = {Hybrid diffusion-optimization for quantum synthesis with continuous native gates},
  author  = {Guo, Dajun and Hu, Chukun and Su, Xiaolu},
  journal = {Physica Scripta},
  year    = {2026},
  doi     = {10.1088/1402-4896/ae6a4a},
  url     = {https://iopscience.iop.org/article/10.1088/1402-4896/ae6a4a}
}
```

---

<!-- ## License
 -->

