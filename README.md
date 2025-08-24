# GPT-2-like LLM from Scratch 🤖
This repository is a work-in-progress project to build a complete GPT-2 like Large Language Model (LLM) from the ground up, with a focus on understanding the core components and architecture. The project is being developed step-by-step, and this README will be updated to reflect the current state of progress.

## Current Progress & Implemented Features ⚙️
The following foundational components of the model have been implemented and tested:
* **Multi-Head Attention:** The core of the Transformer, this module processes input sequences by allowing the model to focus on different parts of the sequence simultaneously. The implementation includes separate learnable weights for queries, keys, and values, and a mechanism for combining the output of multiple attention heads.

* **Feedforward Network with GELU:** A two-layer, fully connected network that processes each position of the sequence independently. This implementation uses the **GELU** *(Gaussian Error Linear Unit)* activation function, which provides a smoother, non-linear transformation compared to traditional functions.

* **Transformer Block Forward Pass:** A complete and functional forward pass for a singular Transformer block has been implemented. This demonstrates the flow of data through the Multi-Head Attention and Feedforward Network, including their respective normalizations and connections.

* **Shortcut Connections:** We have successfully added residual or "shortcut" connections around the attention and feedforward network modules. These connections are crucial for training deep neural networks, as they help mitigate the vanishing gradient problem and enable the flow of information through multiple layers.


## Stay tuned for more updates!