import faiss
import numpy as np
import os
import joblib
from typing import List

class VectorDB:
    def __init__(self):
        pass

    def build_index(self, embeddings: np.ndarray,  labels: List[str], ids: List[str], index_file_path: str, metric=faiss.METRIC_INNER_PRODUCT):
        """
        Build a FAISS index from embeddings and save it to disk, along with the corresponding unique IDs and labels.
        
        Args:
            embeddings (np.ndarray): The embeddings to index.
            ids (List[str]): The list of unique IDs corresponding to each embedding.
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
        
        metadata_dict = {i: {'id': unique_id, 'label': label} for i, (unique_id, label) in enumerate(zip(ids, labels))}
        metadata_dict_path = os.path.join(index_file_path, "metadata_dict.pkl")
        joblib.dump(metadata_dict, metadata_dict_path)

        label_dict = {i: label for i, label in enumerate(labels)}
        label_dict_path = os.path.join(index_file_path, "label_dict.pkl")
        joblib.dump(label_dict, label_dict_path)

        return index_file_path, metadata_dict_path

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


    def load_metadata(self, metadata_file_path: str) -> dict:
        """
        Load the metadata dictionary from disk, containing unique IDs and labels.
        
        Args:
            metadata_file_path (str): The path to the saved metadata dictionary file.
        
        Returns:
            dict: Dictionary of index-to-unique-ID-and-label mappings.
        """
        return joblib.load(metadata_file_path)

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

    def get_metadata_from_indices(self, indices: np.ndarray, metadata_dict: dict) -> List[dict]:
        """
        Retrieve the corresponding unique IDs and labels for the given indices using the metadata dictionary.
        
        Args:
            indices (np.ndarray): Array of nearest neighbor indices.
            metadata_dict (dict): Dictionary of index-to-unique-ID-and-label mappings.
        
        Returns:
            List[dict]: List of dictionaries with 'id' and 'label' for each nearest neighbor.
        """
        return [metadata_dict[i] for i in indices[0]]

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize token embeddings to unit length along last axis.
        
        Args:
            embeddings: Array of shape (..., embedding_dim)
            
        Returns:
            Normalized embeddings with same shape
        """
        # Calculate L2 norm along last axis (embedding dimension)
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, np.finfo(embeddings.dtype).tiny)
        return embeddings / norms
    
    
    def rerank_with_maxsim(self, query_token_embeddings: np.ndarray, top_k_metadata: List[dict], token_embedding_file: str) -> List[dict]:
        """
        Rerank the top-K results using MaxSim based on token-level embeddings from an .npz file.
        
        Args:
            query_token_embeddings (np.ndarray): The token-level embeddings of the query (shape: max_seq_len, embedding_dim).
            top_k_metadata (List[dict]): List of metadata for top-K items from FAISS, each containing 'id' and 'label'.
            token_embedding_file (str): Path to the .npz file containing token-level embeddings.
        
        Returns:
            List[dict]: Reranked list of metadata dictionaries based on MaxSim.
        """
        with np.load(token_embedding_file) as token_embeddings_data:
            maxsim_scores = []
            for item in top_k_metadata:
                unique_id = item['id']
                
                candidate_token_embeddings = token_embeddings_data[unique_id]
                
                candidate_token_embeddings = self.normalize_embeddings(candidate_token_embeddings)
                maxsim = np.max(np.inner(query_token_embeddings, candidate_token_embeddings))
                maxsim_scores.append((item, maxsim))
        
        reranked_metadata = [item for item, _ in sorted(maxsim_scores, key=lambda x: x[1], reverse=True)]
        
        return reranked_metadata

    def majority_vote(self, reranked_metadata: List[dict]) -> str:
        """
        Perform majority voting on the labels of the reranked results.
        
        Args:
            reranked_metadata (List[dict]): List of reranked metadata dictionaries, each containing 'label'.
        
        Returns:
            str: The label with the majority vote.
        """
        label_counts = {}
        for item in reranked_metadata:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return max(label_counts, key=label_counts.get)
    

