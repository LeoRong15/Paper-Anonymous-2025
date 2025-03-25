import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.criterion(logits, labels)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        eps = 1e-8
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        features_norm = features / (torch.norm(features, dim=1, keepdim=True) + eps)
        similarity = torch.matmul(features_norm, features_norm.T) / self.temperature
        similarity = torch.clamp(similarity, min=-1e2, max=1e2)
        exp_similarity = torch.exp(similarity)
        mask_sum = torch.maximum(mask.sum(1), torch.ones_like(mask_sum) * eps)
        loss = -torch.log(exp_similarity / (exp_similarity.sum(1, keepdim=True) + eps) + eps)
        loss = (mask * loss).sum(1) / mask_sum
        return loss.mean()
