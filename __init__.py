import os
import torch
import logging
from fiftyone.operators import types
from .zoo import TorchRadioModelConfig, TorchRadioModel, RadioOutputProcessor, SpatialHeatmapOutputProcessor

logger = logging.getLogger(__name__)

# Model variants and their configurations
MODEL_VARIANTS = {
    "nv_labs/c-radio_v4-h": {
        "model_version": "c-radio_v4-h",
    },
    "nv_labs/c-radio_v4-so400m": {
        "model_version": "c-radio_v4-so400m",
    },
}

def download_model(model_name, model_path):
    """Downloads the model.
    
    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    import os
    import torch
    
    model_info = MODEL_VARIANTS[model_name]
    model_version = model_info["model_version"]
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Downloading RADIO model {model_version}...")
    
    # Load model from torch hub and save to disk
    model = torch.hub.load(
        'NVlabs/RADIO', 
        'radio_model', 
        version=model_version, 
        progress=True, 
        skip_validation=True
    )
    
    # Save the model state dict to the specified path
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_version': model_version,
        'model_config': getattr(model, 'config', None),
        'model_class': model.__class__.__name__,
    }, model_path)
    
    logger.info(f"RADIO model {model_version} saved to {model_path}")


def load_model(
    model_name, 
    model_path, 
    output_type="summary",
    feature_format="NCHW",
    use_external_preprocessor=False,
    **kwargs
):
    """Loads the model.
    
    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            downloaded, as declared by the ``base_filename`` field of the
            manifest
        output_type: what to return - "summary" or "spatial"
        feature_format: "NCHW" or "NLC" for spatial features format
        use_external_preprocessor: whether to use external preprocessing
        **kwargs: additional keyword arguments
        
    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    model_info = MODEL_VARIANTS[model_name]
    model_version = model_info["model_version"]
    
    config_dict = {
        "model_version": model_version,
        "model_path": model_path,  # Add the model path for loading from disk
        "output_type": output_type,
        "feature_format": feature_format,
        "use_external_preprocessor": use_external_preprocessor,
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
    
    inputs.enum(
        "feature_format", 
        ["NCHW", "NLC"],
        default="NCHW",
        label="Feature Format",
        description="Format for spatial features: NCHW (computer vision) or NLC (transformer)"
    )
    
    inputs.bool(
        "use_external_preprocessor",
        default=False,
        label="Use External Preprocessor",
        description="Whether to use external preprocessing (advanced option)"
    )

    inputs.bool(
        "apply_smoothing",
        default=True,
        label="Apply Gaussian Smoothing",
        description="Whether to smooth the spatial heatmaps for better visualization"
    )

    inputs.float(
        "smoothing_sigma",
        default=1.0,
        label="Smoothing Sigma",
        description="The standard deviation (sigma) for Gaussian blur applied to heatmaps"
    )

    return types.Property(inputs)