import os
import logging
from fiftyone.operators import types
from .zoo import TorchRadioModelConfig, TorchRadioModel, RadioOutputProcessor, SpatialHeatmapOutputProcessor

logger = logging.getLogger(__name__)

# Model variants and their Hugging Face repository names
MODEL_VARIANTS = {
    "nv_labs/c-radio_v4-h": {
        "hf_repo": "nvidia/C-RADIOv4-H",
    },
    "nv_labs/c-radio_v4-so400m": {
        "hf_repo": "nvidia/C-RADIOv4-SO400M",
    },
}

def download_model(model_name, model_path):
    """Downloads the model from Hugging Face.
    
    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    from transformers import AutoModel, CLIPImageProcessor
    
    model_info = MODEL_VARIANTS[model_name]
    hf_repo = model_info["hf_repo"]
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Downloading C-RADIOv4 model from Hugging Face: {hf_repo}")
    
    # Download model and processor from Hugging Face
    # This caches them in the HF cache directory
    CLIPImageProcessor.from_pretrained(hf_repo)
    AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
    
    # Write a marker file to indicate the model is downloaded
    # The actual model weights are cached by Hugging Face
    with open(model_path, 'w') as f:
        f.write(f"hf_repo={hf_repo}\n")
    
    logger.info(f"C-RADIOv4 model {hf_repo} downloaded and cached")


def load_model(
    model_name, 
    model_path, 
    output_type="summary",
    **kwargs
):
    """Loads the model from Hugging Face.
    
    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            downloaded, as declared by the ``base_filename`` field of the
            manifest
        output_type: what to return - "summary" or "spatial"
        **kwargs: additional keyword arguments
        
    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    model_info = MODEL_VARIANTS[model_name]
    hf_repo = model_info["hf_repo"]
    
    config_dict = {
        "hf_repo": hf_repo,
        "output_type": output_type,
        "raw_inputs": True,  # We handle preprocessing ourselves
        **kwargs
    }
    
    # Set up output processor based on output type
    if output_type == "summary":
        # For embeddings
        config_dict["as_feature_extractor"] = True
        config_dict["output_processor_cls"] = RadioOutputProcessor
        config_dict["output_processor_args"] = {"output_type": output_type}
    elif output_type == "spatial":
        # For heatmaps
        config_dict["output_processor_cls"] = SpatialHeatmapOutputProcessor
        config_dict["output_processor_args"] = {
            "apply_smoothing": kwargs.get("apply_smoothing", True),
            "smoothing_sigma": kwargs.get("smoothing_sigma", 1.51),
        }
    else:
        raise ValueError(f"Unsupported output_type: {output_type}. Use 'summary' or 'spatial'")
    
    config = TorchRadioModelConfig(config_dict)
    return TorchRadioModel(config)

def resolve_input(model_name, ctx):
    """Defines any necessary properties to collect the model's custom
    parameters from a user during prompting.
    
    Args:
        model_name: the name of the model, as declared by the ``base_name`` and
            optional ``version`` fields of the manifest
        ctx: an :class:`fiftyone.operators.ExecutionContext`
        
    Returns:
        a :class:`fiftyone.operators.types.Property`, or None
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    inputs = types.Object()
    
    inputs.enum(
        "output_type",
        ["summary", "spatial"],
        default="summary",
        label="Output Type",
        description="Type of features to extract: summary (global embeddings) or spatial (heatmaps)"
    )

    inputs.bool(
        "apply_smoothing",
        default=True,
        label="Apply Gaussian Smoothing",
        description="Whether to smooth the spatial heatmaps for better visualization"
    )

    inputs.float(
        "smoothing_sigma",
        default=1.51,
        label="Smoothing Sigma",
        description="The standard deviation (sigma) for Gaussian blur applied to heatmaps"
    )

    inputs.bool(
        "use_mixed_precision",
        default=True,
        label="Use Mixed Precision",
        description="Use bfloat16 mixed precision for faster inference (requires Ampere+ GPU)"
    )

    return types.Property(inputs)