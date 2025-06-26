# Deep Learning Text Generation Project - Complete Overview

## 🎯 Project Purpose

This is a **transformer-based language model** for character-level text generation, supporting both English and Hebrew languages. The project includes training, hyperparameter optimization, and text generation capabilities.

## 📁 File Structure & Purpose

### Core Model Architecture

- **`transformer.py`**: Main transformer implementation

  - `TransformerLM`: Complete language model with embedding, transformer blocks, and output projection
  - `TransformerDecoderBlock`: Individual transformer layer (attention + MLP)
  - `Embed`: Token and positional embedding layer
  - Supports configurable layers, heads, embedding size, residual connections, dropout

- **`attention.py`**: Self-attention mechanism implementation

  - `CausalSelfAttention`: Multi-head causal self-attention with masking
  - Core functions: `attention_scores()`, `self_attention()`, `create_causal_mask()`
  - Handles multiple attention heads and dropout

- **`mlp.py`**: Feed-forward network
  - `MLP`: Simple 2-layer network with GELU activation
  - Used in transformer blocks after attention

### Data Processing

- **`data.py`**: Data loading and tokenization

  - `CharTokenizer`: Character-level tokenizer with vocabulary management
  - `RandomOrderDataIterator`: Training data iterator with random sampling
  - `load_data()`: Main function that caches tokenized data to disk
  - Supports both English and Hebrew datasets

- **`decode-hebrew.py`**: Hebrew text encoding/decoding utilities

### Training & Model Management

- **`main.py`**: Main training script with CLI interface

  - Extensive hyperparameter configuration (layers, heads, embed size, learning rate, etc.)
  - Training loop with loss computation, optimization, checkpointing
  - Text generation with prompts ("JULIET:" for English, "בבוקר אחד" for Hebrew)
  - GPU support and checkpoint resumption

- **`lm.py`**: Language model utilities
  - `batch_to_labeled_samples()`: Prepare input/label pairs for training
  - `compute_loss()`: Cross-entropy loss with padding handling
  - `save_checkpoint()`/`load_checkpoint()`: Model persistence with metadata

### Hyperparameter Optimization

- **`run_search.py`**: Automated hyperparameter search

  - Tests 100+ configurations across multiple GPUs
  - Search space: layers (4,6,8), heads (4,6,8), embed sizes (128,192,256), learning rates, optimizers, dropout, weight decay
  - Adaptive sampling based on previous best results
  - Parallel execution with result tracking

- **`analyze_search_results.py`**: Search result analysis
  - Statistical analysis of hyperparameter performance
  - Parameter importance ranking and performance metrics

### Testing & Data

- **`tests.py`**: Comprehensive unit tests for attention mechanisms
- **`data/input.txt`**: English training data (Shakespeare plays, ~1.1MB)
- **`heb-data/`**: Hebrew training data (Bialik's works, multiple files)

## 🔧 Key Features

### Multi-Language Support

- **English**: Shakespeare plays with "JULIET:" generation prompt
- **Hebrew**: Bialik's works with "בבוקר אחד" generation prompt
- **RTL Display**: Proper Hebrew text rendering with `flip_rtl()` function

### Flexible Architecture

- Configurable transformer parameters (layers, heads, embedding dimensions)
- Optional residual connections
- Multiple initialization methods (Xavier, Kaiming, Normal)
- Configurable dropout and weight decay

### Training Capabilities

- Regular checkpointing with generated samples
- Resume training from checkpoints
- Progress monitoring with loss tracking
- Multi-GPU support

## 🚀 Usage Examples

### Basic Training

```bash
# Train English model
python main.py --hebrew=False --n_layers=6 --n_heads=6 --embed_size=192

# Train Hebrew model
python main.py --hebrew=True --n_layers=6 --n_heads=6 --embed_size=192
```

### Load and Generate

```bash
# Load trained model and generate samples
python main.py --load=checkpoints/english_step_50000 --hebrew=False
```

### Hyperparameter Search

```bash
# Run search on English data
python run_search.py --language=english --gpus=3

# Run search on Hebrew data
python run_search.py --language=hebrew --gpus=3
```

### Analyze Results

```bash
python analyze_search_results.py english
python analyze_search_results.py hebrew
```

## 🧠 Technical Architecture

**Model Type**: Decoder-only transformer (similar to GPT)

**Architecture Flow**:

1. Character-level tokenization → Token embeddings + Positional embeddings
2. Multiple transformer blocks (Causal self-attention + MLP)
3. Layer normalization and optional residual connections
4. Output projection to vocabulary size
5. Next-token prediction for text generation

**Key Components**:

- **Tokenization**: Character-level for both languages
- **Embeddings**: Token + positional embeddings
- **Attention**: Causal self-attention with triangular masking
- **MLP**: Feed-forward networks with GELU activation
- **Training**: Cross-entropy loss with teacher forcing
- **Generation**: Autoregressive sampling with temperature/top-k options

## 📊 Hyperparameter Search Space

**Architecture**: layers [4,6,8], heads [4,6,8], embed_size [128,192,256], mlp_size [512,768,1024]
**Training**: learning_rate [5e-4, 3e-4, 1e-4], optimizer [adamw, sgd], weight_decay [0.0, 0.01, 0.05]
**Regularization**: dropout [0.0, 0.1], residuals [False, True], init [xavier, kaiming, normal]

## 🎯 Project Goals

1. **Educational**: Demonstrate transformer architecture implementation
2. **Research**: Explore hyperparameter optimization for language models
3. **Multilingual**: Support both English and Hebrew text generation
4. **Practical**: Provide working text generation capabilities

## 🔍 Key Implementation Details

- **Causal Masking**: Ensures each token only attends to previous tokens
- **Character-Level**: Simpler vocabulary but longer sequences
- **Checkpointing**: Saves model state, optimizer state, tokenizer, and generated samples
- **RTL Support**: Special handling for Hebrew text display
- **Parallel Search**: Multi-GPU hyperparameter optimization
- **Adaptive Sampling**: Uses previous best results to guide search

This project serves as a complete example of modern transformer implementation for text generation with comprehensive training, optimization, and evaluation capabilities.
