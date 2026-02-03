# NVLabs C-RADIOv4 Models for FiftyOne

![RADIO Models in FiftyOne](cradio-fiftyone.gif)

This repository provides FiftyOne integration for C-RADIOv4 models from NVIDIA Labs. C-RADIOv4 is the latest release in the C-RADIO family of agglomerative vision foundation models, which leverage multi-teacher distillation to create a unified student model that retains and improves the distinct capabilities of multiple teachers.

C-RADIOv4 builds upon AM-RADIO/RADIOv2.5, trained with an updated set of teachers: **SigLIP2**, **DINOv3**, and **SAM3**. This enables strong improvements on key downstream tasks including semantic segmentation, zero-shot classification, and dense perception—all at the same computational complexity as previous versions.

## 🚀 Features

- **Multiple Model Variants**: C-RADIOv4-H (631M params) and C-RADIOv4-SO400M (412M params)
- **Multi-Teacher Distillation**: Combines capabilities of SigLIP2, DINOv3, and SAM3
- **Dual Output Types**: Extract global summary embeddings or spatial attention features
- **Attention Heatmaps**: Visualize what regions the model focuses on (PCA-based visualization)
- **Efficient Batching**: Full support for batch processing with parallel data loading
- **Any-Resolution Support**: Improved resolution scaling from 128px to 1152px+
- **ViTDet Mode**: Windowed attention for dramatically faster inference at high resolutions
- **FiftyOne Integration**: Seamless integration with FiftyOne's computer vision workflows
- **GPU Acceleration**: Optimized CUDA support with automatic mixed precision (bfloat16)
- **Permissive License**: NVIDIA Open Model License Agreement


## 🛠️ Installation

```bash
# Install FiftyOne
pip install fiftyone

# Register the RADIO model source
import fiftyone.zoo as foz
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/CRADIOv4",
)
```

## 🚀 Quick Start

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart", shuffle=True)

# Load RADIO model for embeddings
model = foz.load_zoo_model("nv_labs/c-radio_v4-h")

# Compute embeddings
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings",
)

# Launch FiftyOne App
session = fo.launch_app(dataset)
```

## 📊 Available Models

| Model Name | Parameters | Architecture | Zero-Shot | kNN | ADE20k | Best For |
|------------|------------|--------------|-----------|-----|--------|----------|
| `nv_labs/c-radio_v4-h` | 631M | ViT-H | 83.09 | 86.59 | 55.20 | Maximum accuracy, recommended |
| `nv_labs/c-radio_v4-so400m` | 412M | SO400M | 82.01 | 85.76 | 55.14 | Balanced performance, faster inference |

*Benchmarks: Zero-Shot and kNN accuracy on ImageNet-1K, ADE20k linear probe mIoU. C-RADIOv4 is competitive with DINOv3 on dense tasks at a fraction of the parameters.*

## ⚙️ Model Configuration

### Output Types

```python
# Global image embeddings (default)
model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    output_type="summary"  # Global semantic features
)

# Spatial attention features
spatial_model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h", 
    output_type="spatial"  # Patch-level spatial features
)
```

### Feature Formats

```python
# returns a 1D embedding vector, dimensions 3048
model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    output_type="summary",
    feature_format="NCHW"  # "NCHW": [Batch, Channels, Height, Width] , or you can use "NLC":[Batch, Num_patches, Channels]
)

# returns spatial features which are parsed as a FiftyOne Heatmap
model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    output_type="spatial", 
    feature_format="NCHW" # can only use this format for spatial features
)
```

### Complete Configuration Options

```python
model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    
    # Core settings
    output_type="spatial",              # "summary" or "spatial"
    feature_format="NCHW",             # "NCHW" or "NLC" (NCHW only for spatial)
    
    # Performance options
    use_mixed_precision=True,          # Auto-detected, bfloat16 on Ampere+
    use_external_preprocessor=False,   # Advanced preprocessing
    
    # Spatial heatmap options (when output_type="spatial")
    apply_smoothing=True,              # Smooth attention heatmaps
    smoothing_sigma=1.51,              # Gaussian smoothing strength
    
)
```

## 🔥 Use Cases & Examples

### 1. Global Image Embeddings

Extract high-level semantic representations for similarity search and clustering:

```python
# Compute embeddings with efficient batching
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings",
    batch_size=16,      # Process 16 images per batch
    num_workers=4       # Parallel data loading
)
```

### 2. Spatial Attention Heatmaps

Visualize what regions the model pays attention to (uses PCA visualization as per the C-RADIOv4 paper):

```python
# Load spatial model with smoothing
spatial_model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    output_type="spatial",
    apply_smoothing=True,   # Gaussian smoothing for cleaner heatmaps
    smoothing_sigma=1.51,   # Smoothing strength
    feature_format="NCHW"
)

# Generate attention heatmaps with batching
dataset.apply_model(
    spatial_model, 
    "radio_heatmap",
    batch_size=16,
    num_workers=4
)

# View heatmaps in FiftyOne App
session = fo.launch_app(dataset)
```

### 3. Embedding Visualization with UMAP

Create 2D visualizations of your image embeddings:

```python
import fiftyone.brain as fob

# First compute embeddings
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings"
)

# Create UMAP visualization
results = fob.compute_visualization(
    dataset,
    method="umap",  # Also supports "tsne", "pca"
    brain_key="radio_viz",
    embeddings="radio_embeddings"
)

# Explore in the App
session = fo.launch_app(dataset)
```

### 4. Similarity Search

Build powerful similarity search with RADIO embeddings:

```python
import fiftyone.brain as fob

# Build similarity index
results = fob.compute_similarity(
    dataset,
    backend="sklearn",  # Fast sklearn backend
    brain_key="radio_sim", 
    embeddings="radio_embeddings"
)

# Find similar images
sample_id = dataset.first().id
similar_samples = dataset.sort_by_similarity(
    sample_id,
    brain_key="radio_sim",
    k=10  # Top 10 most similar
)

# View results
session = fo.launch_app(similar_samples)
```

### 5. Dataset Representativeness

Score how representative each sample is of your dataset:

```python
import fiftyone.brain as fob

# Compute representativeness scores
fob.compute_representativeness(
    dataset,
    representativeness_field="radio_represent",
    method="cluster-center",
    embeddings="radio_embeddings"
)

# Find most representative samples
representative_view = dataset.sort_by("radio_represent", reverse=True)
```

### 6. Duplicate Detection

Find and remove near-duplicate images:

```python
import fiftyone.brain as fob

# Detect duplicates using embeddings
results = fob.compute_uniqueness(
    dataset,
    embeddings="radio_embeddings"
)

# Filter to most unique samples
unique_view = dataset.sort_by("uniqueness", reverse=True)

```

### 7. Advanced: Custom Analysis Pipeline

Combine multiple RADIO outputs for comprehensive analysis:

```python
# Step 1: Global embeddings for similarity
embedding_model = foz.load_zoo_model("nv_labs/c-radio_v4-h")
dataset.compute_embeddings(embedding_model, "radio_embeddings")

# Step 2: Spatial heatmaps for attention analysis
spatial_model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    output_type="spatial",
    apply_smoothing=True,
    smoothing_sigma=0.8
)
dataset.apply_model(spatial_model, "radio_heatmap")

# Step 3: Build similarity index
import fiftyone.brain as fob
fob.compute_similarity(dataset, embeddings="radio_embeddings", brain_key="radio_sim")

# Step 4: Comprehensive analysis
session = fo.launch_app(dataset)
```

## 🔧 Model Architecture Details

### C-RADIOv4 Foundation Models
C-RADIOv4 uses multi-teacher distillation to combine the strengths of three state-of-the-art foundation models:

- **SigLIP2**: Enhanced text-image alignment for zero-shot classification
- **DINOv3**: Improved semantic segmentation and dense perception capabilities  
- **SAM3**: Enables replacing SAM3's vision encoder for segmentation tasks

### Key Improvements in v4
- **Stochastic Resolution Training**: Trained across resolutions from 128px to 1152px for smooth resolution scaling
- **Shift Equivariance**: Novel loss formulation that prevents learning fixed-pattern noise from teachers
- **ViTDet Mode**: Optional windowed attention that dramatically reduces inference time at high resolutions
- **Balanced Summary Loss**: Improved angular loss normalization between teachers

### Architecture Variants
| Variant | Parameters | Base Architecture | Notes |
|---------|------------|-------------------|-------|
| C-RADIOv4-H | 631M | ViT-H | Maximum performance |
| C-RADIOv4-SO400M | 412M | SigLIP SO400M | Competitive with ViT-H at lower cost |

### Feature Specifications
- **Spatial Features**: Rich channel features at multiple spatial scales with cleaner object boundaries
- **Resolution Scaling**: Strong performance from 256px to 1536px+ (achieves 57.72 mIoU at 1536px on ADE20k)
- **Preprocessing**: Automatic RGB normalization to [0,1] range
- **Device Management**: Automatic GPU/CPU placement with mixed precision support

## ⚡ Performance Tips

### Batch Processing

For optimal performance on large datasets, use batching with parallel data loading:

```python
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings",
    batch_size=16,      # Adjust based on GPU memory
    num_workers=4       # Parallel data loading workers
)
```

**Recommended batch sizes by GPU:**
- RTX 3090/4090 (24GB): `batch_size=16-32`
- A100 (40GB/80GB): `batch_size=32-64`
- Smaller GPUs: `batch_size=4-8`

## 🛠️ Troubleshooting

### Common Issues

**GPU Memory Errors**
```python
# Use smaller batch size, smaller model, or disable mixed precision
model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-so400m",  # Use smaller model
    use_mixed_precision=False
)

dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings",
    batch_size=4        # Reduce batch size
)
```

**Mixed Precision Issues**
```python
# Disable mixed precision on older GPUs (pre-Ampere)
model = foz.load_zoo_model(
    "nv_labs/c-radio_v4-h",
    use_mixed_precision=False
)
```

## 📖 Citation

```bibtex
@misc{ranzinger2026cradiov4,
      title={C-RADIOv4 (Tech Report)},
      author={Mike Ranzinger and Greg Heinrich and Collin McCarthy and Jan Kautz and Andrew Tao and Bryan Catanzaro and Pavlo Molchanov},
      year={2026},
      eprint={2601.17237},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.17237},
}
```

## 📄 License

C-RADIOv4 is released under the **NVIDIA Open Model License Agreement**, a commercially permissive license. Please refer to the [NVIDIA RADIO repository](https://github.com/NVlabs/RADIO) for complete license details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- 🐛 Bug reports and issues
- 💡 Feature requests and suggestions  
- 🔧 Pull requests with improvements
- 📖 Documentation enhancements

## 🔗 Related Resources

- [NVIDIA RADIO GitHub](https://github.com/NVlabs/RADIO) - Original implementation
- [C-RADIOv4 Models on Hugging Face](https://huggingface.co/nvidia/C-RADIOv4) - Pre-trained models
- [C-RADIOv4 Tech Report](https://arxiv.org/abs/2601.17237) - Technical details and benchmarks
- [FiftyOne Documentation](https://docs.voxel51.com/) - Computer vision workflows
- [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html) - Model ecosystem
- [FiftyOne Brain](https://docs.voxel51.com/user_guide/brain.html) - ML-powered dataset curation