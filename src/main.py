import os
import torch
import random
import numpy as np
from data_utils import load_dataset
from models import EnhancedConvNeXt, enhance_convnext_model
from train_utils import train_model
from visualize import plot_metrics, save_metrics  # 如果移除绘图，则只导入 save_metrics
import torchvision.models as models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    
    image_path = "/root/Proj/dataset/COVID19-4"  # Adjust path as needed
    batch_size = 64
    epochs = 50
    save_path = './train.pth'
    
    train_loader, validate_loader = load_dataset(image_path, batch_size)
    
    model_path = '/root/models/convnext_tiny-983f1562.pth'
    base_model = models.convnext_tiny(pretrained=False)
    base_model.load_state_dict(torch.load(model_path), strict=False)
    net = EnhancedConvNeXt(base_model, num_classes=4)
    net = enhance_convnext_model(net)
    net.to(device)
    
    train_losses, val_metrics, best_metrics = train_model(
        net, train_loader, validate_loader, epochs, device, save_path
    )
    
    plot_metrics(train_losses, val_metrics)
    save_metrics(best_metrics)

if __name__ == "__main__":
    main()
