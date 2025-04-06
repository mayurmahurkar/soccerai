from typing import Generator, Iterable, List, TypeVar, Dict, Tuple, Optional

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import pickle
import cv2

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


def compute_crop_hash(crop: np.ndarray) -> str:
    """
    Compute a simple hash for an image crop to use as a cache key.
    
    Args:
        crop (np.ndarray): Image crop
        
    Returns:
        str: A hash string representing the image
    """
    # Resize to small dimensions to make hashing faster
    small_img = cv2.resize(crop, (32, 32))
    # Convert to grayscale to reduce dimensionality
    if small_img.ndim == 3:
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    # Create hash from pixel values
    return str(hash(small_img.tobytes()))


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32, use_cache: bool = True):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
           use_cache (bool): Whether to cache extracted features.
       """
        self.device = device
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        self.feature_cache: Dict[str, np.ndarray] = {}

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        if not self.use_cache:
            return self._extract_features_no_cache(crops)
        
        # Use cache to avoid redundant feature extraction
        cache_hits = []
        crops_to_process = []
        cache_indices = []
        
        for i, crop in enumerate(crops):
            crop_hash = compute_crop_hash(crop)
            if crop_hash in self.feature_cache:
                cache_hits.append(self.feature_cache[crop_hash])
                cache_indices.append(i)
            else:
                crops_to_process.append((i, crop, crop_hash))
        
        # Extract features for crops not in cache
        if crops_to_process:
            indices = [item[0] for item in crops_to_process]
            uncached_crops = [item[1] for item in crops_to_process]
            crop_hashes = [item[2] for item in crops_to_process]
            
            features = self._extract_features_no_cache(uncached_crops)
            
            # Update cache with new features
            for i, (idx, feature) in enumerate(zip(indices, features)):
                self.feature_cache[crop_hashes[i]] = feature
        
        # Combine cached and newly extracted features in the original order
        if cache_hits and crops_to_process:
            # Create a combined array with all features in the original order
            all_features = np.zeros((len(crops), cache_hits[0].shape[0]), dtype=np.float32)
            
            # Place cached features
            for i, feature in zip(cache_indices, cache_hits):
                all_features[i] = feature
                
            # Place newly extracted features
            for (orig_idx, _, _), feature in zip(crops_to_process, features):
                all_features[orig_idx] = feature
                
            return all_features
        elif cache_hits:
            return np.vstack(cache_hits)
        elif crops_to_process:
            return features
        else:
            return np.array([])

    def _extract_features_no_cache(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features without using cache.
        
        Args:
            crops (List[np.ndarray]): List of image crops.
            
        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            # for batch in tqdm(batches, desc='Embedding extraction'):
            for batch in batches:
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data) if data else np.array([])

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)
        
    def predict_batch(self, all_crops: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Predict the cluster labels for multiple batches of image crops.
        
        Args:
            all_crops (List[List[np.ndarray]]): List of batches of image crops.
            
        Returns:
            List[np.ndarray]: Predicted cluster labels for each batch.
        """
        # Flatten all crops for efficient batch processing
        flat_crops = []
        crop_indices = []
        start_idx = 0
        
        for batch in all_crops:
            flat_crops.extend(batch)
            batch_size = len(batch)
            crop_indices.append((start_idx, start_idx + batch_size))
            start_idx += batch_size
            
        if not flat_crops:
            return [np.array([]) for _ in all_crops]
            
        # Extract features for all crops at once
        all_data = self.extract_features(flat_crops)
        all_projections = self.reducer.transform(all_data)
        all_predictions = self.cluster_model.predict(all_projections)
        
        # Split predictions back into original batches
        results = []
        for start, end in crop_indices:
            results.append(all_predictions[start:end])
            
        return results

    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        # Clear the cache before saving to reduce file size
        cache_state = self.use_cache
        cache_content = self.feature_cache
        
        self.use_cache = False
        self.feature_cache = {}
        
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
            
        # Restore cache settings
        self.use_cache = cache_state
        self.feature_cache = cache_content

    @classmethod
    def load(cls, filepath: str) -> 'TeamClassifier':
        """Load the model from a file."""
        with open(filepath, 'rb') as file:
            classifier = pickle.load(file)
            # Initialize empty cache for loaded model if not present
            if not hasattr(classifier, 'feature_cache'):
                classifier.feature_cache = {}
            if not hasattr(classifier, 'use_cache'):
                classifier.use_cache = True
            return classifier