import os
import numpy as np
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from setfit import SetFitModel
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from scipy.spatial.distance import cosine

def cosine_distance(x, y):
    return cosine(x, y)

class OutlierDetector:

    def __init__(self, contamination: float, output_dir: str, n_neighbors: int = 20):
        """
        Initialize the OutlierDetector with a SetFit model and a LOF model for outlier detection.

        Args:
            setfit_model (SetFitModel): Pretrained SetFit model body to use for embedding extraction.
            contamination (float): Proportion of outliers in the data, between 0 and 1.
            output_dir (str): Directory where the model's components will be saved.
            n_neighbors (int): Number of neighbors to use for LOF (default is 20).
        """
        self.outlier_detector_lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="cosine", contamination=contamination, novelty=True)
        self.outlier_detector_if = IsolationForest(contamination=contamination, random_state=42)
        self.output_dir = output_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_outliers(self, embeddings, train_dataset):
        self._train_isolation_forest(embeddings)
        self._train_lof(embeddings)
        self._visualize_embeddings(embeddings, train_dataset['label'])

    def _train_isolation_forest(self, embeddings):
        """Train the Isolation Forest on embeddings extracted from the SetFit model."""
        self.logger.info("Training IF model...")
        self.outlier_detector_if.fit(embeddings)

    def _train_lof(self, embeddings):
        """Train the Local Outlier Factor model on embeddings extracted from the SetFit model."""
        self.logger.info("Training LOF model...")
        self.outlier_detector_lof.fit(embeddings)

    def _visualize_embeddings(self, embeddings: np.ndarray, labels: list):
        """Visualize embeddings using t-SNE and save the plot as a file."""
        self.logger.info("Visualizing embeddings...")
        
        n_samples, n_features = embeddings.shape
        max_components = min(n_samples, n_features)
        n_components_pca = min(50, max_components)
        pca = PCA(n_components=n_components_pca)
        self.logger.info(f"Using {n_components_pca} components for PCA (max allowed: {max_components})")

        reduced_embeddings = pca.fit_transform(embeddings)

        
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne.fit_transform(reduced_embeddings)

        plt.figure(figsize=(10, 8))
        
        for label in set(labels):
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=label)

            cluster_center = tsne_embeddings[indices].mean(axis=0)
            plt.text(cluster_center[0], cluster_center[1], label, fontsize=10, weight='normal',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        plt.title(f"t-SNE Visualization of Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

        output_file = os.path.join(self.output_dir, f"tsne_embeddings_plot.png")
        plt.savefig(output_file)
        self.logger.info(f"t-SNE plot saved to {output_file}")


    def save_outliers(self):
        self._save_lof_model()
        self._save_isolation_forest()

    def _save_lof_model(self):
        """Since LOF cannot be used for prediction after training, we only save the embeddings if needed."""
        output_file = os.path.join(self.output_dir, "model_lof_outlier_head.pkl")
        joblib.dump(self.outlier_detector_lof, output_file)
        self.logger.info(f"LOF OOS detection model saved to {output_file}")

    def _save_isolation_forest(self):
        """Save the trained Isolation Forest model as a pickle file in the output directory."""
        output_file = os.path.join(self.output_dir, "model_if_outlier_head.pkl")
        joblib.dump(self.outlier_detector_if, output_file)
        self.logger.info(f"IF OOS detection model saved to {output_file}")
        

