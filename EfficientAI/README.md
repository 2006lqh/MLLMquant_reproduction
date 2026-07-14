# 🚀 EfficientAI

> **Efficient Inference for LLMs & MLLMs**  
> An open-source research project from Alibaba Cloud dedicated to efficient large language model inference.

<p align="center">
  <img src="./images/banner_2.png" alt="EfficientAI Banner" width="600" style="max-width: 100%; height: auto;">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://arxiv.org/search/EfficientAI"><img src="https://img.shields.io/badge/Papers-3-green" alt="Papers"></a>
  <a href="https://github.com/alibaba/EfficientAI/stargazers"><img src="https://img.shields.io/github/stars/alibaba/EfficientAI?style=social" alt="Stars"></a>
  <a href="https://github.com/alibaba/EfficientAI/issues"><img src="https://img.shields.io/github/issues/alibaba/EfficientAI" alt="Issues"></a>
</p>

---

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🔥 Latest Updates](#-latest-updates)
- [📦 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🧪 Benchmarks](#-benchmarks)
- [📚 Publications](#-publications)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [✉️ Contact](#-contact)

---

## ✨ Key Features

EfficientAI focuses on **inference-time optimizations** for LLMs and MLLMs:

| Feature | Description | Status |
|---------|-------------|--------|
| 🔹 **Activation Sparsity** | Dynamic sparsity methods for faster inference | ✅ LaRoSa (ICML 2025) |
| 🔹 **Quantization** | Post-training & quantization-aware techniques for MLLMs | ✅ MASQuant (CVPR 2026) |
| 🔹 **Agentic Reasoning** | Efficient tool-use and reasoning frameworks | ✅ D-CORE ( ICML 2026) |
| 🔹 **Reproducible Benchmarks** | Standardized eval pipelines for research & production | 🔄 In Progress |

---

## 🔥 Latest Updates

<details open>
<summary><b>📰 Changelog (Click to expand)</b></summary>

- **[2026-05]** 🚀 **D-CORE** accepted to **ICML 2026**
  → Efficient tool-use reasoning via dynamic computation routing  
  [📄 Paper](https://arxiv.org/abs/2602.02160) | [💻 Code](./dcore/) | [🎮 Demo](#)

- **[2026-03]** 🎉 **MASQuant** accepted to **CVPR 2026**  
  → Multimodal LLM PTQ algorithm with SOTA accuracy-efficiency tradeoff  
  [📄 Paper](https://arxiv.org/abs/2603.04800) | [💻 Code](./masquant/)


- **[2026-01]** 🏆 **LaRoSa** accepted to **ICML 2025**  
  → Training-free activation sparsity for LLM acceleration  
  [📄 Paper](https://arxiv.org/abs/2507.01299) | [💻 Code](./larosa/)

</details>

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/alibaba/EfficientAI.git
cd EfficientAI

# Install dependencies (recommended: use conda)
pip install -r requirements.txt

# Optional: Install with specific module support
# pip install -e ".[larosa]"   # for LaRoSa
# pip install -e ".[masquant]" # for MASQuant
