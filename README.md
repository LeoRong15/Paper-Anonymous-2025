# Anonymous Research Project (2025)

This project implements an enhanced ConvNeXt model with attention mechanisms for COVID19-4 image classification, targeting four categories related to COVID-19 imagery. The enhancements aim to improve feature extraction and classification robustness.

## Project Overview

This project is the open-source implementation of a 2025 research paper, proposing an enhanced ConvNeXt-based convolutional neural network integrated with attention mechanisms for COVID19-4 image classification tasks. By incorporating a attention module into the original ConvNeXt architecture and optimizing the loss function design, this approach improves classification accuracy and adaptability to complex medical imagery.

## Project Structure

- `data/`: Contains the image dataset for training (compressed as `images.zip`).

- ```
  src/
  ```

  - `data_utils.py`: Data loading and preprocessing logic.
  - `models.py`: Defines the enhanced ConvNeXt model with attention mechanisms.
  - `losses.py`: Custom loss function implementations.
  - `train_utils.py`: Training and validation logic.
  - `visualize.py`: Saves and visualizes training metrics.
  - `main.py`: Main script integrating the training pipeline.

- `requirements.txt`: Project dependencies.

- `README.md`: This file.

- `LICENSE`: License agreement.

## Environment Requirements

- Python 3.8+

- Recommended setup with Conda:

  ```bash
  conda create -n myenv python=3.8
  conda activate myenv
  pip install -r requirements.txt
  ```

## Data Preparation

- **Dataset Source**: The dataset used for this project is the "COVID-19 Radiography Database" available on Kaggle. Download it from the following link:
  - [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)

- **Data Format**: After downloading, the dataset contains images organized into category-specific subfolders (e.g., `COVID/`, `Normal/`, `Viral Pneumonia/`, `Lung_Opacity/`).

- **Preparation Steps**:

  1. Download the dataset from the Kaggle link above.
  2. Extract the downloaded archive (e.g., `COVID-19_Radiography_Dataset.zip`) into the `data/` folder:
     ```bash
     unzip COVID-19_Radiography_Dataset.zip -d data/
     ```

  3. Verify the extracted directory structure: `data/[category_name]/[image_files]` (e.g., `data/COVID/`, `data/Normal/`, etc.).

  4. Ensure the folder structure aligns with the four categories expected by the COVID19-4 classification task.

  5. Preprocessing: `data_utils.py` applies normalization by default. Adjust parameters in `data_utils.py` if needed to match the dataset's characteristics.

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/LeoRong15/Paper-Anonymous-2025.git
   cd Paper-Anonymous-2025
   ```

2. **Extract the Data** (as described above):

   - Download the "COVID-19 Radiography Database" from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data).
   - Extract the dataset into the data/ folder:

   ```bash
   unzip COVID-19_Radiography_Dataset.zip -d data/
   ```

3. **Run Training**:

   ```bash
   python src/main.py --data_path data/ --batch_size 32 --epochs 50
   ```

   - Optional Parameters:
     - `--data_path`: Dataset root directory (default: `data/`).
     - `--batch_size`: Batch size (default: 32).
     - `--epochs`: Number of training epochs (default: 50).
     - `--learning_rate`: Learning rate (default: 0.001).
   - See `src/main.py` for additional parameters in the `argparse` configuration.

## Output Results

- `train.pth`: Trained model weights file, suitable for inference or transfer learning.

- `training_results.txt`: Logs training metrics, including:

  - Training and validation loss per epoch.
  - Classification accuracy (Accuracy).
  - F1 Score (useful for evaluating imbalanced datasets).

- Visualization: Generate loss and accuracy curves by running:

  ```bash
  python src/visualize.py
  ```
