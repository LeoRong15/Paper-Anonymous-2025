import os
import pickle
import json
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def get_data_transforms():
    """定义数据预处理和增强的转换操作"""
    return {
        "train": transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

def load_dataset(image_path, batch_size=64):
    """加载和划分数据集"""
    assert os.path.exists(image_path), f"{image_path} does not exist."
    
    data_transform = get_data_transforms()
    full_dataset_train = datasets.ImageFolder(root=image_path, transform=data_transform["train"])
    full_dataset_val = datasets.ImageFolder(root=image_path, transform=data_transform["val"])
    
    train_indices_path = "train_indices.pkl"
    val_indices_path = "val_indices.pkl"
    
    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path):
        print("Loading existing train/val indices...")
        with open(train_indices_path, "rb") as f:
            train_indices = pickle.load(f)
        with open(val_indices_path, "rb") as f:
            val_indices = pickle.load(f)
    else:
        print("Splitting dataset and saving indices...")
        train_indices, val_indices = train_test_split(
            list(range(len(full_dataset_train))), test_size=0.2, random_state=42
        )
        with open(train_indices_path, "wb") as f:
            pickle.dump(train_indices, f)
        with open(val_indices_path, "wb") as f:
            pickle.dump(val_indices, f)
    
    train_dataset = Subset(full_dataset_train, train_indices)
    validate_dataset = Subset(full_dataset_val, val_indices)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    
    print(f"Using {len(train_dataset)} images for training, {len(validate_dataset)} for validation.")
    
    output_file = 'class_indices.json'
    if not os.path.exists(output_file):
        print(f"Creating {output_file}...")
        cla_dict = dict((val, key) for key, val in full_dataset_train.class_to_idx.items())
        with open(output_file, 'w') as json_file:
            json_file.write(json.dumps(cla_dict, indent=4))
    
    return train_loader, validate_loader
