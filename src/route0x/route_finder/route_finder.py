import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, AddedToken
import onnxruntime as ort
import joblib
from .vector_db import VectorDB
from collections import Counter
import os
import logging
import sys

class RouteFinder:
    def __init__(self, 
                 model_dir: str = None, 
                 oos_label: str = "NO_NODES_DETECTED",
                 use_calibrated_head = False,
                 max_length: int = 32,
                 use_compressed_model: bool = False):
        
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path(model_dir)
        self.tokenizer = self._get_tokenizer(max_length)
        
        if os.path.exists(os.path.join(self.model_dir, "model_body.onnx")):
            self.onnx_session = ort.InferenceSession(os.path.join(self.model_dir, "model_body.onnx"))
        else:
            self.logger.info(f"Model file model_body.onnx missing")
            sys.exit(0)    
            
        if use_calibrated_head:    
            self.logger.info(f"Loading calibrated classifier")
            self.classifier = self._load_calibrated_classifier()
        else:    
            self.logger.info(f"Loading un-calibrated classifier")
            self.classifier = self._load_classifier()
        
        self.outlier_detectors = self._load_outlier_detectors()
        self.pooling_strategy = self._get_pooling_strategy()
        self.vectordb = VectorDB()
        self.vec_index = self._load_vector_index()
        self.labels_dict = self.vectordb.load_labels(str(self.model_dir / "label_dict.pkl"))
        self.oos_label = oos_label
        self.max_length = max_length

        # TODO: Default ONNX FP32 offers best latency, only mem usage is the concern, Quantisation might address it but perfomance drop is quite a bit
        # So use_compressed_model at the moment will be discouraged for users, but can be used.
        # if use_compressed_model and os.path.exists(os.path.join(self.model_dir, "model_body_quantized.onnx")):
        #     self.onnx_session = ort.InferenceSession(os.path.join(self.model_dir, "model_body_quantized.onnx"))
        # else:
            # TODO: Expand support for fp16 models, Suprisingly FP16 ONNX actually offers worse latency, not sure why inspite of saving memory


    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """Initializes and configures the tokenizer with padding and truncation."""
        config = json.load(open(str(self.model_dir / "config.json")))
        tokenizer_config = json.load(open(str(self.model_dir / "tokenizer_config.json")))
        tokens_map = json.load(open(str(self.model_dir / "special_tokens_map.json")))
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        return tokenizer

    def _get_pooling_strategy(self) -> str:
        """Read the pooling strategy from the config file located in the Pooling folder."""
        pooling_config_path = self.model_dir / "1_Pooling" / "config.json"
        with open(pooling_config_path, 'r') as config_file:
            pooling_config = json.load(config_file)
        return "mean" if pooling_config.get("pooling_mode_mean_tokens") else "cls"
    
    def _load_vector_index(self):
        """Load the faiss index to memory"""
        return self.vectordb.load_index(str(self.model_dir / "faiss_index.bin"))
    
    def _load_calibrated_classifier(self):
        """Load the calibrated classifier from the stored pickle file."""
        return joblib.load(str(self.model_dir / "model_head_calibrated.pkl"))  
    
    def _load_classifier(self):
        """Load the classifier from the stored pickle file."""
        return joblib.load(str(self.model_dir / "model_head.pkl"))

    def _load_outlier_detectors(self):
        """Load the LOF outlier detector from the stored pickle file."""
        outlier_detectors = []
        outlier_detectors.append(joblib.load(str(self.model_dir / "model_lof_outlier_head.pkl")))
        outlier_detectors.append(joblib.load(str(self.model_dir / "model_if_outlier_head.pkl")))
        return outlier_detectors
    
    def _cls_pooling(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Extract the CLS token embedding (assumed to be the first token)."""
        return token_embeddings[:, 0, :]

    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Perform mean pooling on token embeddings, taking the attention mask into account."""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _get_embeddings(self, text: str) -> np.ndarray:
        """Use the ONNX model to get embeddings from the text input."""
        encoded_input = self.tokenizer.encode(text)
        input_ids = np.array([encoded_input.ids], dtype=np.int64) 
        attention_mask = np.array([encoded_input.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoded_input.type_ids], dtype=np.int64)

        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        onnx_output = self.onnx_session.run(None, onnx_inputs)
        if self.pooling_strategy == "mean":
            embeddings = self._mean_pooling(onnx_output[0], attention_mask)
        elif self.pooling_strategy == "cls":
            embeddings = self._cls_pooling(onnx_output[0])
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

        embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / np.clip(embeddings_norm, a_min=1e-12, a_max=None)

        return embeddings_normalized
        
    def _is_outlier(self, embeddings, use_ensemble=False):
        lof_prediction, lof_threshold_breach = self._predict_with_threshold(self.outlier_detectors[0], embeddings)
        
        if use_ensemble:
            raise NotImplementedError("Ensemble Outlier detection is not yet implemented.")
        else:
            if lof_prediction == 1:
                return False, lof_threshold_breach
            else:
                return True, lof_threshold_breach
            
    def _predict_with_threshold(self, outlier_detector, embeddings):
        prediction = outlier_detector.predict(embeddings)[0]
        raw_score = outlier_detector.decision_function(embeddings)[0]
        
        return prediction, raw_score  
    

    def find_route(self, 
                   query: str,
                   return_raw_scores = False,
                   # Valid only when return_raw_scores = False
                   model_confidence_threshold_for_using_outlier_head: float = 0.9, # If classifier predicts with >=0.9, we take it as-is
                   model_uncertainity_threshold_for_using_nn: float = 0.5,  # If classifier predicts with <= 0.5, we replace it with Majority voted NN  
                   nn_for_fallback: int = 5,
                   ) -> dict:

        embeddings = self._get_embeddings(query)

        is_outlier, _ = self._is_outlier(embeddings)
        route = {"is_oos": is_outlier}

        probabilities = self.classifier.predict_proba(embeddings)[0]
        class_id = np.argmax(probabilities)
        class_name = self.classifier.classes_[class_id]

        route.update({
            "query": query,
            "route_id": class_id, 
            "route_name": class_name, 
            "prob": np.round(probabilities[class_id],2)
        })

        indices, distances = self.vectordb.search_index(embeddings, self.vec_index , num_neighbors=nn_for_fallback)
        vec_nns = self.vectordb.get_labels_from_indices(indices, self.labels_dict)
        label_counts = Counter(vec_nns)
        most_common_label = label_counts.most_common(1)[0][0]  

        common_label_indices = [i for i, label in enumerate(vec_nns) if label == most_common_label]
        mean_distance_for_most_common_label = np.mean([distances[0][i] for i in common_label_indices])

        route.update({
            "majority_voted_route": most_common_label,
            "mean_distance_from_majority_route": mean_distance_for_most_common_label
        })

        if not return_raw_scores:

            predicted_route = route["route_name"]
            prob = float(route['prob'])

            if route['is_oos']:
                if prob < model_confidence_threshold_for_using_outlier_head:
                    predicted_route = self.oos_label
            else:
                if prob <= model_uncertainity_threshold_for_using_nn:
                    predicted_route = route["majority_voted_route"]
                    prob = route["mean_distance_from_majority_route"]

            route["route_name"] =  predicted_route
            route["prob"] =  np.round(prob,2)

        return route                       
