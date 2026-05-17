# Calligraphy SigLIP Model

A custom implementation of SigLIP (Sigmoid Loss for Language Image Pre-Training) optimized for calligraphy text recognition. Built from scratch using PyTorch with depthwise separable convolutions for image encoding and a transformer for text encoding.

## 🎯 Overview

This project demonstrates end-to-end deep learning for calligraphy recognition:

- **Custom Image Encoder**: Efficient depthwise separable convolutions designed for calligraphy visual features
- **Transformer Text Encoder**: Modern transformer architecture for text representation learning
- **SigLIP Training**: Sigmoid loss-based contrastive learning for image-text alignment
- **Real-World Data**: Trained on both synthetic and authentic handwritten calligraphy

## 📊 Dataset
I used synthetic data using hand picked fonts and visualized them to have more data with how much little real data i have.
| Source | Count |
|--------|-------|
| Synthetic (Font Visualization) | 11,578 |
| Handwritten Calligraphy | 2,250 |
| **Total** | **13,300** |
