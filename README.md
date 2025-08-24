# GPT-2-like LLM from Scratch 🤖
This repository is a work-in-progress project to build a complete GPT-2 like Large Language Model (LLM) from the ground up, with a focus on understanding the core components and architecture. The project is being developed step-by-step, and this README will be updated to reflect the current state of progress.

## Current Progress & Implemented Features ⚙️
The following foundational components of the model have been implemented and tested:
* **Multi-Head Attention:** The core of the Transformer, this module processes input sequences by allowing the model to focus on different parts of the sequence simultaneously. The implementation includes separate learnable weights for queries, keys, and values, and a mechanism for combining the output of multiple attention heads.

* **Feedforward Network with GELU:** A two-layer, fully connected network that processes each position of the sequence independently. This implementation uses the **GELU** *(Gaussian Error Linear Unit)* activation function, which provides a smoother, non-linear transformation compared to traditional functions.

* **Transformer Block Forward Pass:** A complete and functional forward pass for a singular Transformer block has been implemented. This demonstrates the flow of data through the Multi-Head Attention and Feedforward Network, including their respective normalizations and connections.

* **Shortcut Connections:** We have successfully added residual or "shortcut" connections around the attention and feedforward network modules. These connections are crucial for training deep neural networks, as they help mitigate the vanishing gradient problem and enable the flow of information through multiple layers.

* **Stacked Transformer Blocks:**  
  Instead of using a single Transformer block, the model now supports stacking multiple blocks on top of each other. Each block processes the sequence outputs from the previous layer, allowing the network to build progressively richer and more abstract representations of text. This stacking is a key ingredient in scaling models like GPT-2, as deeper architectures enable more powerful pattern recognition and improved language understanding.

* **Layer Normalization Strategies:**  
  Added flexibility in how layer normalization is applied inside the Transformer. Two strategies were explored:
  - **Post-Norm:** The original Transformer design, where normalization is applied after the residual connection.  
  - **Pre-Norm:** A more stable variant, where normalization is applied before the attention and feedforward sublayers.  

  Pre-Norm tends to improve gradient flow and makes training deep Transformer networks more stable, while Post-Norm stays faithful to the earliest Transformer architectures. Both options are now supported in the implementation.

* **GPT Model Assembly:**  
  All the components have been integrated into a complete GPT-like model capable of autoregressive text generation. The model includes:
  - **Token embeddings** to represent input text numerically.  
  - **Positional encodings** to provide sequence order information.  
  - **A stack of Transformer blocks** for deep contextual learning.  
  - **An output projection layer** mapping hidden states back to vocabulary logits for next-token prediction.  

   At this stage, the model is **untrained** and initialized with random weights. As a result, any generated text is gibberish rather than meaningful language. The focus so far has been on implementing the architecture correctly. Training the model to produce coherent text will be the subject of the next stage of the project.


## Stay tuned for more updates!