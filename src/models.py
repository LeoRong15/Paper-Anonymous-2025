import torch
import torch.nn as nn
import torchvision.models as models

class ECAAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).unsqueeze(1)
        y = self.conv(y).squeeze(1)
        y = self.sigmoid(y).unsqueeze(2).unsqueeze(3)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super(SpatialAttention, self).__init__()
        self.attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=ks, padding=ks//2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ) for ks in kernel_sizes
        ])
        self.fusion_weights = nn.Parameter(torch.ones(len(kernel_sizes)))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attentions = [module(attention) for module in self.attention_modules]
        weights = torch.softmax(self.fusion_weights, dim=0)
        combined = sum(w * att for w, att in zip(weights, attentions))
        return x * combined + x

class CombinedAttentionBlock(nn.Module):
    def __init__(self, channels, eca_gamma=2, eca_b=1, spa_kernel_size=[3, 5, 7]):
        super(CombinedAttentionBlock, self).__init__()
        self.eca = ECAAttention(channels, eca_gamma, eca_b)
        self.spa = SpatialAttention(spa_kernel_size)

    def forward(self, x):
        x = self.eca(x)
        x = self.spa(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=512, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        proj = self.projection(x)
        norm = torch.maximum(torch.norm(proj, dim=1, keepdim=True), torch.ones_like(proj) * 1e-8)
        return proj / norm

class EnhancedConvNeXt(nn.Module):
    def __init__(self, base_model, num_classes):
        super(EnhancedConvNeXt, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.fc_in_features = 768
        self.norm = nn.LayerNorm(self.fc_in_features)
        self.transition = nn.Sequential(
            nn.Linear(self.fc_in_features, self.fc_in_features),
            nn.LayerNorm(self.fc_in_features),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(self.fc_in_features, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
        self.projection_head = ProjectionHead(self.fc_in_features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        with torch.set_grad_enabled(True):
            features = self.features(x)
            if torch.isnan(features).any():
                features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            pooled = self.avg_pool(features).view(pooled.size(0), -1)
            pooled = self.norm(pooled)
            trans = self.transition(pooled)
            pooled = pooled + trans
            logits = self.classifier(pooled)
            proj_features = self.projection_head(pooled)
            if torch.isnan(logits).any() or torch.isnan(proj_features).any():
                logits = torch.nan_to_num(logits, nan=0.0)
                proj_features = torch.nan_to_num(proj_features, nan=0.0)
            return logits, proj_features

def enhance_convnext_block(block, channels):
    attention = CombinedAttentionBlock(channels)
    layers = list(block.children())
    layers.append(attention)
    return nn.Sequential(*layers)

def enhance_convnext_model(model):
    features = model.features[0]
    stage_channels = {3: 384, 4: 768}
    stage3 = features[5]
    if isinstance(stage3, models.convnext.CNBlock):
        features[5] = enhance_convnext_block(stage3, stage_channels[3])
    elif isinstance(stage3, nn.Sequential):
        for i in range(len(stage3)):
            if isinstance(stage3[i], models.convnext.CNBlock):
                stage3[i] = enhance_convnext_block(stage3[i], stage_channels[3])
    stage4 = features[7]
    if isinstance(stage4, models.convnext.CNBlock):
        features[7] = enhance_convnext_block(stage4, stage_channels[4])
    elif isinstance(stage4, nn.Sequential):
        for i in range(len(stage4)):
            if isinstance(stage4[i], models.convnext.CNBlock):
                stage4[i] = enhance_convnext_block(stage4[i], stage_channels[4])
    return model
