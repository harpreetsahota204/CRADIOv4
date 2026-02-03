import logging
import warnings

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.decomposition import PCA

import fiftyone.core.labels as fol
import fiftyone.core.models as fom
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
import fiftyone.utils.torch as fout
from fiftyone.utils.torch import GetItem

logger = logging.getLogger(__name__)


class RadioGetItem(GetItem):
    """GetItem transform for loading images for RADIO model.
    
    This class handles data loading in DataLoader workers (parallel I/O).
    Keep this lightweight - only do I/O, no heavy computation.
    """
    
    @property
    def required_keys(self):
        """Return list of fields needed from each sample."""
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Load and return a single image.
        
        This runs in DataLoader worker processes (parallel).
        
        Args:
            sample_dict: Dict containing 'filepath' key
            
        Returns:
            PIL Image in RGB format
        """
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        return image


class TorchRadioModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`TorchRadioModel`.

    Args:
        hf_repo: Hugging Face repository name (e.g., "nvidia/C-RADIOv4-H")
        output_type: what to return - "summary" or "spatial"
        use_mixed_precision: whether to use bfloat16 mixed precision
        apply_smoothing: whether to apply Gaussian smoothing to heatmaps
        smoothing_sigma: sigma for Gaussian smoothing
    """

    def __init__(self, d):
        super().__init__(d)

        self.hf_repo = self.parse_string(d, "hf_repo", default="nvidia/C-RADIOv4-H")
        self.output_type = self.parse_string(d, "output_type", default="summary")
        self.use_mixed_precision = self.parse_bool(d, "use_mixed_precision", default=True)
        self.apply_smoothing = self.parse_bool(d, "apply_smoothing", default=True)
        self.smoothing_sigma = self.parse_number(d, "smoothing_sigma", default=1.51)


class TorchRadioModel(fout.TorchImageModel, SupportsGetItem, TorchModelMixin):
    """Wrapper for C-RADIOv4 models from Hugging Face.

    This model supports efficient batching via FiftyOne's SupportsGetItem pattern.
    Use with dataset.apply_model() for optimized batch processing.

    Args:
        config: a :class:`TorchRadioModelConfig`
    """

    def __init__(self, config):
        super().__init__(config)
        
        # REQUIRED for SupportsGetItem: Initialize base class
        SupportsGetItem.__init__(self)
        
        # REQUIRED: Set preprocess flag (GetItem handles preprocessing)
        self._preprocess = False
        
        # Load the RADIO model and image processor from Hugging Face
        self._radio_model, self._image_processor = self._load_radio_model()
        
        # Check and cache mixed precision support
        self._mixed_precision_supported = self._check_mixed_precision_support()

    # ============ FROM Model BASE CLASS ============
    
    @property
    def media_type(self):
        """The media type processed by the model."""
        return "image"
    
    @property
    def transforms(self):
        """Preprocessing transforms.
        
        For SupportsGetItem models, preprocessing happens in GetItem,
        so return None.
        """
        return None
    
    @property
    def preprocess(self):
        """Whether model should apply preprocessing."""
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Allow FiftyOne to control preprocessing."""
        self._preprocess = value
    
    @property
    def ragged_batches(self):
        """Whether this model supports batches with varying sizes.
        
        MUST return False to enable batching, even though our inputs
        (PIL Images) have variable sizes. We handle variable sizes
        via custom collate_fn.
        
        ragged_batches=True would disable batching entirely!
        """
        return False
    
    # ============ FROM SupportsGetItem ============
    
    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for data loading.
        
        Args:
            field_mapping: Optional dict mapping required_keys to dataset fields
            
        Returns:
            RadioGetItem instance for loading images
        """
        return RadioGetItem(field_mapping=field_mapping)
    
    # ============ FROM TorchModelMixin (for variable sizes) ============
    
    @property
    def has_collate_fn(self):
        """Whether this model provides custom batch collation.
        
        Return True because we need custom collation for variable-size images.
        """
        return True
    
    @property
    def collate_fn(self):
        """Custom collation function for batching.
        
        Returns a function that keeps batch as list of PIL Images
        without stacking (which would fail for variable sizes).
        """
        @staticmethod
        def identity_collate(batch):
            """Return batch as-is without stacking."""
            return batch
        return identity_collate

    # ============ INFERENCE METHODS ============
    
    def predict(self, arg):
        """Process a single image.
        
        Args:
            arg: PIL Image, numpy array, or filepath
            
        Returns:
            Prediction result (embedding or heatmap depending on config)
        """
        # Handle filepath input
        if isinstance(arg, str):
            arg = Image.open(arg).convert("RGB")
        elif isinstance(arg, np.ndarray):
            arg = Image.fromarray(arg)
        
        results = self._predict_all([arg])
        return results[0]
            
    def _check_mixed_precision_support(self):
        """Check if the current GPU supports mixed precision with bfloat16."""
        if not self._using_gpu:
            return False
            
        try:
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(self._device)
                # bfloat16 is supported on Ampere (8.0+) and newer architectures
                return device_capability[0] >= 8
            return False
        except Exception as e:
            logger.warning(f"Could not determine mixed precision support: {e}")
            return False

    def _load_radio_model(self):
        """Load the RADIO model from Hugging Face."""
        from transformers import AutoModel, CLIPImageProcessor
        
        hf_repo = self.config.hf_repo
        logger.info(f"Loading C-RADIOv4 model from Hugging Face: {hf_repo}")
        
        # Load image processor and model
        image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
        model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        
        # Move to device and set to eval mode
        model = model.to(self._device)
        model.eval()
        
        return model, image_processor

    def _predict_all(self, imgs):
        """Process a batch of images.
        
        Overrides TorchImageModel._predict_all() to handle RADIO model.
        
        Args:
            imgs: List of PIL Images from GetItem/collate_fn
            
        Returns:
            List of predictions (embeddings or heatmaps depending on config)
        """
        # Store original image sizes for output processing
        original_sizes = [img.size for img in imgs]  # PIL Image.size is (width, height)
        
        # Process each image individually (CLIPImageProcessor handles resizing)
        summaries = []
        spatial_features_list = []
        
        for img in imgs:
            # Preprocess using CLIPImageProcessor
            pixel_values = self._image_processor(
                images=img, 
                return_tensors='pt', 
                do_resize=True
            ).pixel_values
            pixel_values = pixel_values.to(self._device)
            
            # Forward pass with optional mixed precision
            if self.config.use_mixed_precision and self._mixed_precision_supported and self._using_gpu:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        summary, spatial = self._radio_model(pixel_values)
            else:
                with torch.no_grad():
                    summary, spatial = self._radio_model(pixel_values)
            
            summaries.append(summary)
            spatial_features_list.append(spatial)
        
        # Return based on output type
        if self.config.output_type == "summary":
            # Summaries are fixed-size 1D vectors, can be stacked
            batch_summary = torch.cat(summaries, dim=0)
            
            if self._output_processor is not None:
                return self._output_processor(
                    batch_summary, original_sizes, confidence_thresh=self.config.confidence_thresh
                )
            
            return [batch_summary[i].detach().cpu().numpy() for i in range(len(imgs))]
        
        elif self.config.output_type == "spatial":
            # Spatial features have variable H×W, process individually
            if self._output_processor is not None:
                results = []
                for i, spatial in enumerate(spatial_features_list):
                    result = self._output_processor(
                        spatial, [original_sizes[i]], confidence_thresh=self.config.confidence_thresh
                    )
                    results.append(result[0])
                return results
            
            return [spatial.squeeze(0).detach().cpu().numpy() for spatial in spatial_features_list]
        
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")


class RadioOutputProcessor(fout.OutputProcessor):
    """Output processor for RADIO models that handles embeddings output."""
    
    def __init__(self, output_type="summary", **kwargs):
        super().__init__(**kwargs)
        self.output_type = output_type
        
    def __call__(self, output, frame_size, confidence_thresh=None):
        """Process RADIO model output into embeddings.
        
        Args:
            output: tensor from RADIO model
            frame_size: (width, height) - not used for embeddings
            confidence_thresh: not used for embeddings
            
        Returns:
            list of numpy arrays containing embeddings
        """
        batch_size = output.shape[0]
        return [output[i].detach().cpu().numpy() for i in range(batch_size)]


class SpatialHeatmapOutputProcessor(fout.OutputProcessor):
    """Spatial heatmap processor for RADIO using PCA visualization (as per C-RADIOv4 paper)."""

    def __init__(self, apply_smoothing=True, smoothing_sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.apply_smoothing = apply_smoothing
        self.smoothing_sigma = smoothing_sigma

    def __call__(self, output, frame_sizes, confidence_thresh=None):
        """
        Args:
            output: torch.Tensor of shape [B, C, H, W] or [B, N, C] (NLC format)
            frame_sizes: list of (width, height) for each image
            confidence_thresh: unused

        Returns:
            List of fol.Heatmap instances
        """
        batch_size = output.shape[0]
        heatmaps = []

        for i in range(batch_size):
            spatial = output[i].detach().cpu().numpy()
            
            # Handle different feature formats
            if spatial.ndim == 2:  # [N, C] - NLC format without batch
                # Assume square spatial grid
                N, C = spatial.shape
                H = W = int(np.sqrt(N))
                if H * W != N:
                    # Non-square, find closest factors
                    for h in range(int(np.sqrt(N)), 0, -1):
                        if N % h == 0:
                            H, W = h, N // h
                            break
                spatial = spatial.reshape(H, W, C).transpose(2, 0, 1)  # [C, H, W]
            
            C, H, W = spatial.shape

            # Handle NaN/Inf values (C-RADIOv4 paper notes models can have noise patches)
            spatial = np.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip outliers using percentiles (handles extreme noise patches)
            p_low, p_high = np.percentile(spatial, [1, 99])
            spatial = np.clip(spatial, p_low, p_high)

            # Flatten spatial grid to [H*W, C] for PCA (as used in paper figures)
            reshaped = spatial.reshape(C, -1).T  # [H*W, C]

            # PCA to reduce channels to 1D attention per pixel
            pca = PCA(n_components=1)
            attention_1d = pca.fit_transform(reshaped).reshape(H, W)

            # Optional smoothing
            if self.apply_smoothing:
                attention_1d = gaussian_filter(attention_1d, sigma=self.smoothing_sigma)

            # Resize to match original image dimensions
            orig_w, orig_h = frame_sizes[i]
            attention_resized = resize(
                attention_1d,
                (orig_h, orig_w),
                preserve_range=True,
                anti_aliasing=True
            )

            # Normalize to uint8 [0, 255]
            att_min, att_max = attention_resized.min(), attention_resized.max()
            if att_max > att_min:
                attention_uint8 = ((attention_resized - att_min) / (att_max - att_min) * 255).astype(np.uint8)
            else:
                attention_uint8 = np.zeros_like(attention_resized, dtype=np.uint8)

            heatmap = fol.Heatmap(
                map=attention_uint8,
                range=[0, 255]
            )
            heatmaps.append(heatmap)

        return heatmaps
