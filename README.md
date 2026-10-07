# All-in-One Image Restoration

> **A unified deep learning framework for restoring images affected by multiple types of degradation using degradation-aware prompting and difference-based feature learning.**

This project presents a **unified model for multi-degradation image restoration**, designed to handle different degradation types within a single restoration framework rather than relying on separate models for each degradation.

The proposed architecture combines **128-dimensional degradation prompts** with **difference-based features** and employs a **dual encoder-decoder architecture with mid-level transformation** to improve contextual understanding of degradation characteristics and guide the restoration process.

The model achieved **34.5 dB PSNR** and **0.97 SSIM**, while demonstrating **faster convergence than state-of-the-art approaches such as PromptIR** in our experiments.

---

## 📌 Overview

Image restoration aims to recover a high-quality image from an input affected by degradations such as noise, rain, haze, blur, and other distortions.

A major challenge in restoration is the fact that different degradations require different restoration behaviors. Conventional approaches often train separate networks for individual degradation types, while more recent **all-in-one restoration** methods attempt to use a single model for multiple degradations.

This project explores a unified restoration architecture that explicitly incorporates information about the degradation into the restoration process.

### Key idea

Instead of depending only on the degraded image itself, the proposed framework combines:

- **Degradation-aware 128D prompts**
- **Difference-based feature representations**
- **Dual encoder-decoder architecture**
- **Mid-level feature transformation**
- **Context-aware restoration**

These components allow the network to capture both the **image content** and the **characteristics of the degradation**, enabling a single model to adapt its restoration behavior across different degradation conditions.

---

## 🎯 Objectives

The main objectives of this research project are:

1. Develop a **single unified model** capable of handling multiple image degradation types.
2. Encode degradation information using **128-dimensional prompts**.
3. Enhance feature representation using **difference-based features**.
4. Introduce a **dual encoder-decoder architecture** for richer multi-scale representation learning.
5. Use **mid-level transformation** to improve contextual understanding of degradation characteristics.
6. Achieve high-quality reconstruction while maintaining efficient convergence.
7. Compare the proposed approach against existing all-in-one restoration methods such as **PromptIR**.

---

## 🧠 Proposed Method

The proposed architecture consists of several complementary components.

### 1. Degradation-Aware Prompting

A **128-dimensional prompt representation** is used to encode information related to the degradation present in the input image.

The prompt acts as a compact representation of the degradation characteristics and provides additional information to the restoration network.

Conceptually:

```text
Degraded Image
      │
      ▼
Feature Extraction
      │
      ▼
Degradation Representation
      │
      ▼
128D Prompt
```

The resulting prompt is incorporated into the restoration pipeline so that the network can adapt its feature processing according to the degradation characteristics.

---

### 2. Difference-Based Feature Extraction

Along with conventional image features, the architecture incorporates **difference-based feature representations**.

Difference operations emphasize changes and local variations within feature maps, allowing the network to capture:

- Fine image structures
- Local intensity variations
- Edges and texture transitions
- Degradation-specific patterns
- High-frequency information

This improves the representation of information that can be lost or weakened by standard convolution-based feature extraction.

The difference-based features are subsequently fused with the degradation-aware representations to provide the restoration network with richer information.

---

### 3. Dual Encoder-Decoder Architecture

The proposed model employs a **dual encoder-decoder design**.

The encoder stages progressively extract hierarchical representations from the degraded input, while the decoder stages reconstruct the restored image.

A simplified view of the architecture is:

```text
                     ┌──────────────────────┐
                     │    Degraded Image    │
                     └──────────┬───────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
          Main Feature Encoder        Difference Feature Path
                  │                           │
                  └─────────────┬─────────────┘
                                │
                                ▼
                     128D Degradation Prompt
                                │
                                ▼
                    Mid-Level Transformation
                                │
                                ▼
                     Feature Interaction/Fusion
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
             Decoder Branch 1            Decoder Branch 2
                  │                           │
                  └─────────────┬─────────────┘
                                │
                                ▼
                       Restored Image
```

The dual-path design helps maintain complementary information during restoration while allowing degradation-aware features to influence the reconstruction process.

---

### 4. Mid-Level Transformation

A **mid-level transformation module** is introduced between the encoder and decoder stages.

Rather than performing restoration using only low-level image features, this stage transforms and integrates intermediate representations so that the network can better understand the relationship between:

- Image content
- Degradation characteristics
- Structural information
- Contextual information

This provides a stronger representation before the decoding and reconstruction stages.

---

## 🔬 Research Motivation

Many traditional image restoration methods are **task-specific**.

For example:

```text
Noisy Image  ──► Denoising Model
Rainy Image  ──► Deraining Model
Hazy Image   ──► Dehazing Model
Blurry Image ──► Deblurring Model
```

This requires multiple independently trained models.

An all-in-one restoration framework instead aims to learn:

```text
                 ┌────────────────────┐
                 │   Degraded Image   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Unified Restoration│
                 │       Model        │
                 └─────────┬──────────┘
                           │
                           ▼
                  Restored Clean Image
```

The challenge is that the network must understand **what type of degradation is present** and adapt the restoration process accordingly.

The proposed model addresses this challenge by explicitly introducing **degradation-aware prompts** alongside enhanced feature representations.

---

## ⚙️ Training Pipeline

The overall training process can be summarized as:

```text
Input Degraded Image
        │
        ▼
Pre-processing / Augmentation
        │
        ▼
Feature Extraction
        │
        ├──────────────► Difference-Based Features
        │
        ▼
Degradation Representation
        │
        ▼
128D Prompt Generation
        │
        ▼
Prompt + Image Feature Fusion
        │
        ▼
Dual Encoder
        │
        ▼
Mid-Level Transformation
        │
        ▼
Dual Decoder
        │
        ▼
Restored Image
        │
        ▼
Comparison with Ground Truth
        │
        ▼
Loss Optimization
```

---

## 🧪 Data Processing

The repository includes a feature-extraction/data-loading notebook containing experiments on the **Rain100L** dataset.

The implemented data pipeline includes:

- Paired degraded and clean images
- RGB image loading
- Random **256 × 256** crops during training
- Random rotations of **0°, 90°, 180°, and 270°**
- Tensor conversion using `torchvision`
- Separate training and testing image subsets

The notebook uses PyTorch data loaders together with PIL, NumPy, Torchvision, and Albumentations for image preparation.

---

## 📊 Results

The proposed model achieved the following image restoration performance:

| Metric | Proposed Model |
|--------|----------------|
| **PSNR** | **34.5 dB** |
| **SSIM** | **0.97** |

### Performance Highlights

- **PSNR:** 34.5 dB
- **SSIM:** 0.97
- Unified restoration framework for multiple degradation types
- **Faster convergence** than state-of-the-art approaches such as **PromptIR**

PSNR measures pixel-level reconstruction quality, while SSIM evaluates structural similarity between the restored and reference images.

Higher values for both metrics indicate better restoration quality.

---

## ⚡ Comparison with PromptIR

**PromptIR** introduced a prompt-based framework for all-in-one blind image restoration, using degradation-aware prompts to guide restoration across multiple degradation types.

This project builds on the general motivation of degradation-aware restoration while exploring a different feature-learning strategy by combining:

```text
128D Degradation Prompts
            +
Difference-Based Features
            +
Dual Encoder-Decoder
            +
Mid-Level Transformation
            ↓
     Unified Restoration
```

In our experiments, the proposed model achieved **34.5 dB PSNR** and **0.97 SSIM**, with **faster convergence than PromptIR**.

---

## 🔍 Why Difference-Based Features?

Standard convolution primarily aggregates information from neighboring pixels/features.

Difference-based feature extraction provides an additional representation of local changes and high-frequency information.

This is particularly useful for restoration because degradation often manifests through changes in:

- Edges
- Textures
- Local intensity
- High-frequency structures
- Fine image details

Difference-based representations therefore complement conventional deep features and provide additional information to the reconstruction network.

The idea is related to the use of difference-enhanced convolution for improving feature representation in restoration-oriented networks such as DEA-Net.

---

## 🏗️ Architecture Summary

The major components of the proposed model are:

| Component | Purpose |
|-----------|---------|
| **128D Prompt** | Encodes degradation-specific information |
| **Difference-Based Features** | Captures local changes and high-frequency information |
| **Dual Encoder-Decoder** | Learns complementary hierarchical representations |
| **Mid-Level Transformation** | Improves contextual understanding of degradation |
| **Feature Fusion** | Combines degradation-aware and image-content features |
| **Decoder** | Reconstructs the high-quality image |

---

## 📁 Repository Structure

```text
ECC-351-Final-project/
│
├── All_in_one_image_restoration.pdf
│       └── Project report describing the proposed architecture,
│           methodology, experiments and results
│
├── feature-extraction.ipynb
│       └── Feature extraction and dataset preparation experiments
│
└── README.md
        └── Project documentation
```

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **PIL / Pillow**
- **Albumentations**
- **Jupyter Notebook**

---

## 📈 Evaluation Metrics

### PSNR

**Peak Signal-to-Noise Ratio (PSNR)** measures the reconstruction fidelity between the restored image and the ground-truth image.

```text
PSNR = 10 × log10(MAX² / MSE)
```

Higher PSNR generally indicates lower reconstruction error.

### SSIM

**Structural Similarity Index (SSIM)** measures the structural similarity between the restored image and the reference image.

It evaluates characteristics such as:

- Luminance
- Contrast
- Structure

A value closer to **1.0** indicates stronger structural similarity.

---

## 📚 Related Work

### PromptIR

**PromptIR: Prompting for All-in-One Blind Image Restoration**

PromptIR introduced a prompt-based approach for all-in-one image restoration, where degradation-specific prompts dynamically guide the restoration network across different degradation types and levels.

Repository:  
https://github.com/va1shn9v/PromptIR

Paper:  
https://arxiv.org/abs/2306.13090

### DEA-Net

**DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention**

DEA-Net explores detail-enhanced convolution and content-guided attention for improving image restoration and introduces difference-based feature enhancement within convolutional representations.

Repository:  
https://github.com/cecret3350/DEA-Net

Paper:  
https://arxiv.org/abs/2301.04805

---

## 🚀 Key Contributions

The project can be summarized through the following contributions:

### 1. Unified Multi-Degradation Restoration
Designed a **single restoration framework** capable of learning across multiple degradation conditions.

### 2. 128D Degradation Prompts
Introduced **128-dimensional prompts** to provide explicit degradation-aware information to the restoration network.

### 3. Difference-Based Feature Learning
Fused degradation-aware prompts with **difference-based feature representations** to capture complementary local and high-frequency information.

### 4. Dual Encoder-Decoder
Employed a **dual encoder-decoder architecture** to learn richer hierarchical image representations.

### 5. Mid-Level Contextual Transformation
Introduced **mid-level transformation** to improve the network's contextual understanding of degradation characteristics.

### 6. Improved Convergence
Achieved **faster convergence than PromptIR** while obtaining strong restoration quality.

### 7. Quantitative Performance
Achieved:

```text
PSNR  = 34.5 dB
SSIM  = 0.97
```

---

## 📖 Project Report

The detailed research report is available in:

**[All_in_one_image_restoration.pdf](./All_in_one_image_restoration.pdf)**

It contains the detailed methodology, architectural design, experimental analysis, and results.

---

## 🔗 Repository

GitHub Repository:

https://github.com/ARPIT-27-PANDEY/ECC-351-Final-project

---

## 📜 References

1. Potlapalli, V., Zamir, S. W., Khan, S., & Khan, F. S.  
   **PromptIR: Prompting for All-in-One Blind Image Restoration.**  
   NeurIPS 2023.  
   https://arxiv.org/abs/2306.13090

2. Chen, Z., He, Z., & Lu, Z.-M.  
   **DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention.**  
   IEEE Transactions on Image Processing, 2024.  
   https://arxiv.org/abs/2301.04805

---

## 👨‍💻 Author

**Arpit Kumar Pandey**  
Final Year, Electronics & Communication Engineering  
Indian Institute of Technology Roorkee

GitHub:  
https://github.com/ARPIT-27-PANDEY
