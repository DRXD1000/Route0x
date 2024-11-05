import faiss
import numpy as np
import os
import joblib
from typing import List

class VectorDB:

    def __init__(self):
        pass

    def build_index(self, embeddings: np.ndarray, labels: List[str], index_file_path: str, metric=faiss.METRIC_INNER_PRODUCT):
        """
        Build a FAISS index from embeddings and save it to disk, along with the corresponding labels.
        
        Args:
            embeddings (np.ndarray): The embeddings to index.
            labels (List[str]): The list of text labels corresponding to each embedding.
            index_file_path (str): The path where the index file will be saved.
            metric: FAISS metric for distance calculation (e.g., inner product for cosine similarity).
        
        Returns:
            str: Path to the saved index file.
        """
        dimension = embeddings.shape[1]

        if metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexHNSWFlat(dimension, 32) 
        elif metric == faiss.METRIC_L2:
            index = faiss.IndexHNSWFlat(dimension, 32) 
        else:
            raise ValueError("Unsupported metric. Choose between METRIC_INNER_PRODUCT and METRIC_L2.")

        index.add(embeddings)
        faiss.write_index(index, os.path.join(index_file_path, "faiss_index.bin"))
        label_dict = {i: label for i, label in enumerate(labels)}
        label_dict_path = os.path.join(index_file_path, "label_dict.pkl")
        joblib.dump(label_dict, label_dict_path)
        
        return index_file_path, label_dict_path

    def load_index(self, index_file_path: str) -> faiss.Index:
        """
        Load a FAISS index from disk.
        
        Args:
            index_file_path (str): The path to the saved FAISS index file.
        
        Returns:
            faiss.Index: The loaded FAISS index.
        """
        index = faiss.read_index(index_file_path)
        return index

    def load_labels(self, label_file_path: str) -> dict:
        """
        Load the label dictionary from disk.
        
        Args:
            label_file_path (str): The path to the saved label dictionary file.
        
        Returns:
            dict: Dictionary of index-to-label mappings.
        """
        return joblib.load(label_file_path)

    def search_index(self, query_embedding: np.ndarray, index: faiss.Index, num_neighbors=5):
        """
        Search for the nearest neighbors using the FAISS index.
        
        Args:
            query_embedding (np.ndarray): The embedding of the query.
            index (faiss.Index): The FAISS index to search in.
            num_neighbors (int): The number of nearest neighbors to retrieve.
        
        Returns:
            np.ndarray: Indices of the nearest neighbors.
            np.ndarray: Distances to the nearest neighbors.
        """
        distances, indices = index.search(query_embedding, num_neighbors)
        return indices, distances

    def get_labels_from_indices(self, indices: np.ndarray, label_dict: dict) -> List[str]:
        """
        Retrieve the corresponding labels for the given indices using the label dictionary.
        
        Args:
            indices (np.ndarray): Array of nearest neighbor indices.
            label_dict (dict): Dictionary of index-to-label mappings.
        
        Returns:
            List[str]: List of corresponding labels for the nearest neighbors.
        """
        return [label_dict[i] for i in indices[0]]  
