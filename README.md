# Transformer from Scratch
This repository contains a PyTorch implementation of a Transformer model from scratch.

### Running the Model
To train and evaluate the model, simply run:
``` bash
python main.py
```

### How to Use
1. Clone the repository:
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd transformer-from-scratch
```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Run the main script:
```bash
python main.py
```

### Overview
This implementation includes:

- A `MultiHeadAttention` mechanism
- A `PositionWiseFeedForward` network
- Positional encoding
- Transformer blocks
- Utility functions for training and evaluating the model
- A dataset loader for testing purposes

### Explanation
***Multi-Head Attention:***
This module allows the model to jointly attend to information from different representation subspaces.

***Position-Wise Feed-Forward Network:***
A fully connected feed-forward network applied to each position separately and identically.

***Positional Encoding:***
Since the transformer contains no recurrence and no convolution, to make use of the order of the sequence, we inject some information about the relative or absolute position of the tokens in the sequence.

***Transformer Block:***
Combines the multi-head attention mechanism and the position-wise feed-forward network.

***Transformer Model:***
Stacks several transformer blocks to form the complete model.

### Acknowledgements
This project is inspired by the original Transformer paper "Attention is All You Need" by Vaswani et al.
