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



## Model Architecture

Both encoders are implemented in `siglip_modules.py`:

**Image Encoder**
- Input: Greyscale images in shape `(B, 1, H, W)` where B is batch size
- Output: Image embeddings

**Text Encoder**
- Input: Text sequences in shape `(B, L)` where L is sequence length
- Text is mapped using the vocabulary mapper from `datasetimg.np` (access via `mapper` key)
- Output: Text embeddings

## Pre-trained Weights

Model weights are saved in `SIGLIP.pt` as a PyTorch dictionary:

```python
checkpoint = torch.load('SIGLIP.pt')

# Access components
image_encoder_weights = checkpoint['Iencoder']
text_encoder_weights = checkpoint['Tencoder']
temperature = checkpoint['t_prime']  # learned temperature parameter
bias = checkpoint['b']  # learned bias parameter
optimizer_state = checkpoint['opti']  # for resuming training
config = checkpoint['config']  # model configs (Iencoder, Tencoder)
```
## Results/Performance

Just keep it simple. Add something like:

```markdown
## Performance

The model successfully learns to align calligraphy images with their text representations, trained on:
- 11k synthetic calligraphy images (font visualization)
- 2.3k real handwritten samples

The learned embeddings enable:
- Image-to-text retrieval
- Text-to-image matching
- Calligraphy text recognition
- Image-to-Image similarity

```
## Installation

Clone the repository:
```bash
git clone https://github.com/beastreader/caligraphy-siglip-module.git
cd caligraphy-siglip-module
```


## Contact

Built by [@beastreader](https://github.com/beastreader)
- Email: mahmoudgalal621@gmail.com
