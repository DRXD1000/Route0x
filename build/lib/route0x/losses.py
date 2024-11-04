import torch
import torch.nn.functional as F
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import CoSENTLoss
import math
    
class PairwiseArcFaceFocalLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 0.5, scale: float = 64, focal_gamma: float = 2.0, focal_weight: float = 0.5):
        super().__init__()
        self.model = model
        self.margin = margin
        self.scale = scale
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.focal_gamma = focal_gamma  
        self.focal_weight = focal_weight  

    def focal_loss(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * bce_loss).mean()
        return focal_loss    

    def forward(self, sentence_features, labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        
        embeddings_a = embeddings[0]
        embeddings_b = embeddings[1]

        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        cosine = F.cosine_similarity(embeddings_a, embeddings_b, dim=1)

        positive_pairs = labels == 1
        negative_pairs = labels == 0

        cos_theta_m = cosine[positive_pairs] * self.cos_margin - torch.sqrt(1.0 - torch.pow(cosine[positive_pairs], 2)) * self.sin_margin
        cos_theta_m = torch.where(cosine[positive_pairs] > self.th, cos_theta_m, cosine[positive_pairs] - self.mm)

        scaled_positive = self.scale * cos_theta_m
        scaled_negative = self.scale * cosine[negative_pairs]

        positive_labels = torch.ones_like(scaled_positive, dtype=torch.long)
        negative_labels = torch.zeros_like(scaled_negative, dtype=torch.long)

        logits = torch.cat([scaled_positive, scaled_negative], dim=0)
        targets = torch.cat([positive_labels, negative_labels], dim=0)

        # loss = F.cross_entropy(logits.unsqueeze(1), targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        focal_loss = self.focal_loss(logits, targets)
        loss = bce_loss + self.focal_weight * focal_loss
        
        return loss


class BinaryLabelTripletMarginLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 1.0):
        super().__init__()
        self.model = model
        self.margin = margin

    def forward(self, sentence_features, labels):
        
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        embeddings_a = embeddings[0]  
        embeddings_b = embeddings[1]  

        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        positive_mask = labels == 1
        
        if not torch.any(positive_mask) or not torch.any(~positive_mask):
            return torch.tensor(0.0, requires_grad=True).to(embeddings_a.device)

        anchors = embeddings_a[positive_mask]
        positives = embeddings_b[positive_mask]
        
        available_negatives = embeddings_b[~positive_mask]
        
        if len(available_negatives) < len(anchors):
            num_repeats = (len(anchors) // len(available_negatives)) + 1
            available_negatives = available_negatives.repeat(num_repeats, 1)
            
        negatives = available_negatives[:len(anchors)]
        
        loss = F.triplet_margin_loss(
            anchor=anchors,
            positive=positives,
            negative=negatives,
            margin=self.margin
        )
        
        return loss


class ScaledAnglELoss(CoSENTLoss):
    def __init__(self, model: SentenceTransformer, scale: float = 64.0) -> None:
        super().__init__(model, scale, similarity_fct=util.pairwise_angle_sim)


class LMCLLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 0.35, scale: float = 64):
        super().__init__()
        self.model = model
        self.margin = margin
        self.scale = scale

    def forward(self, sentence_features, labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        embeddings_a = embeddings[0]
        embeddings_b = embeddings[1]

        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        cosine_sim = F.cosine_similarity(embeddings_a, embeddings_b)

        positive_margin = (labels == 1).float() * self.margin
        cosine_sim_with_margin = cosine_sim - positive_margin

        scaled_cosine_sim = self.scale * cosine_sim_with_margin

        log_softmax = F.log_softmax(scaled_cosine_sim, dim=0)

        int_labels = labels.long()

        loss = F.mse_loss(log_softmax, int_labels.float()) 

        return loss

