import torch
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score
import numpy as np

def train_model(net, train_loader, validate_loader, epochs, device, save_path):
    classification_loss = ClassificationLoss()
    contrastive_loss = ContrastiveLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    best_metrics = {'acc': 0.0, 'recall': 0.0, 'f1': 0.0, 'precision': 0.0, 'kappa': 0.0}
    train_losses = []
    val_metrics = {'accuracies': [], 'recalls': [], 'f1s': [], 'precisions': [], 'kappas': []}
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, proj_features = net(images)
                cls_loss = classification_loss(logits, labels)
                cont_loss = contrastive_loss(proj_features, labels)
                cont_weight = 0.1 * (1 - math.exp(-epoch))
                total_loss = cls_loss + cont_weight * cont_loss
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += total_loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{total_loss:.3f}"
        
        net.eval()
        acc = 0.0
        all_predict_y, all_val_labels = [], []
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs, _ = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
                all_predict_y.extend(predict_y.cpu().numpy())
                all_val_labels.extend(val_labels.cpu().numpy())
        
        val_accurate = acc / len(validate_loader.dataset)
        val_predict_y, val_true_y = np.array(all_predict_y), np.array(all_val_labels)
        val_recall = recall_score(val_true_y, val_predict_y, average='macro', zero_division=0)
        val_f1 = f1_score(val_true_y, val_predict_y, average='macro', zero_division=0)
        val_precision = precision_score(val_true_y, val_predict_y, average='macro', zero_division=0)
        val_kappa = cohen_kappa_score(val_true_y, val_predict_y)
        
        print(f'[epoch {epoch + 1}] train_loss: {running_loss/len(train_loader):.4f} '
              f'val_accuracy: {val_accurate:.4f} val_recall: {val_recall:.4f} '
              f'val_f1: {val_f1:.4f} val_precision: {val_precision:.4f} val_kappa: {val_kappa:.4f}')
        
        if val_accurate > best_metrics['acc']:
            best_metrics.update({'acc': val_accurate, 'recall': val_recall, 'f1': val_f1,
                                 'precision': val_precision, 'kappa': val_kappa})
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': val_accurate,
            }, save_path)
            print(f"New best model saved! Accuracy: {val_accurate:.4f}")
        
        train_losses.append(running_loss / len(train_loader))
        val_metrics['accuracies'].append(val_accurate)
        val_metrics['recalls'].append(val_recall)
        val_metrics['f1s'].append(val_f1)
        val_metrics['precisions'].append(val_precision)
        val_metrics['kappas'].append(val_kappa)
        scheduler.step()
    
    return train_losses, val_metrics, best_metrics
