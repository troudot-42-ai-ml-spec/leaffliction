# Leaffliction 🍃: Plant Disease Classification with Deep Learning

This project, part of the 42 school curriculum, focuses on plant disease classification using deep learning techniques. It implements a comprehensive computer vision pipeline for detecting and classifying leaf diseases using TensorFlow and various image processing tools.

## 📊 Project Overview

Leaffliction is a complete machine learning solution that combines data augmentation, data transformation, model training, and inference capabilities to identify plant diseases from leaf images. The project currently supports classification of 8 different disease categories across Apple and Grape plants.

### Supported Disease Classes:

- **Apple**: Black rot, Healthy, Rust, Scab
- **Grape**: Black rot, Esca, Healthy, Leaf spot

### Key Features:

- **Data Augmentation**: Advanced image augmentation pipeline using albumentations
- **Image Transformation**: Comprehensive image processing pipeline using PlantCV
- **Model Training**: CNN-based deep learning model with TensorFlow/Keras
- **Dataset Analysis**: Distribution visualization and statistical analysis tools
- **Prediction**: Real-time inference on new leaf images
- **Model Persistence**: Save and load trained models with datasets

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip or uv (Python package manager)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/troudot/leaffliction.git
   cd leaffliction
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Or using uv:

   ```bash
   uv sync
   ```

3. Freeze the environment before a push (for contributors):

   ```bash
   uv lock # Freeze the uv.lock file

   # Freeze requirements.txt for pip installation from uv.lock
   uv pip compile uv.lock --output-file requirements.txt
   # or from pyproject.toml
   uv pip compile pyproject.toml --output-file requirements.txt
   ```

## 🔧 Usage

### 1. Data Analysis and Visualization

Visualize the distribution of your dataset:

```bash
python srcs/Distribution.py /path/to/dataset
```

### 2. Image Augmentation

Generate and visualize augmented versions of a single image:

```bash
python srcs/Augmentation.py /path/to/image.jpg
```

### 3. Image Transformation

Apply image processing transformations using PlantCV:

```bash
python srcs/Transformation.py --ops operation1,operation2 --src /path/to/images --dst /path/to/output
```

### 4. Model Training

Train a new model on your dataset:

```bash
python srcs/train.py /path/to/training/dataset
```

The training script will:

- Load and split your dataset (70% train, 20% validation, 10% test)
- Apply data augmentation to the training set
- Train a CNN model for the specified number of epochs
- Save the trained model and datasets to `model.zip`

### 5. Prediction

Make predictions on new images using a trained model:

```bash
python srcs/predict.py /path/to/leaf/image.jpg
```

This will output predictions sorted by confidence for all supported disease classes.

## 📁 Project Structure

```
leaffliction/
│
├── srcs/
│   ├── models/                 # Model architectures (empty - models defined in utils)
│   ├── pipeline/              # Image transformation pipelines
│   │   └── transforms_pipeline.py
│   ├── transforms/            # Custom image transforms
│   ├── utils/                 # Utility functions
│   │   ├── parsing/           # Data parsing utilities
│   │   ├── plotting/          # Visualization tools
│   │   ├── augmentation.py    # Data augmentation functions
│   │   ├── build_model.py     # Model architecture definition
│   │   ├── hyperparams.py     # Training hyperparameters
│   │   └── train_model.py     # Training loop implementation
│   ├── Augmentation.py        # Image augmentation script
│   ├── Distribution.py        # Dataset distribution analysis
│   ├── Transformation.py      # Image transformation pipeline
│   ├── predict.py            # Inference script
│   └── train.py              # Training script
├── train_set/                # Training dataset directory
├── test_set/                 # Test dataset directory
├── models/                   # Saved model files
├── main.py                   # Main entry point
├── pyproject.toml           # Project configuration
├── uv.lock                  # Lock file for uv sync
└── requirements.txt         # Dependencies (if using pip)
```

## 🛠️ Configuration

### Hyperparameters

Default training parameters can be found in `srcs/utils/hyperparams.py`:

```python
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
CLASSES = 8
EPOCHS = 20
```

### Data Augmentation

The project uses albumentations for advanced image augmentation techniques including:

- Random rotations and flips
- Color space transformations
- Noise injection
- Geometric distortions

## 🧪 Model Architecture

The project implements a Convolutional Neural Network (CNN) optimized for leaf disease classification. The model architecture includes:

- Convolutional layers with ReLU activation
- Max pooling for spatial dimension reduction
- Batch normalization for training stability
- Dropout layers for regularization
- Dense layers for final classification

## 📈 Performance

The model achieves competitive performance on the supported plant disease datasets. Training metrics including accuracy, loss, and validation performance are tracked during the training process.

## 📝 Code Quality

The project maintains high code quality standards using:

- **Black**: Code formatting
- **Flake8**: Linting and style checking
- **Type hints**: For better code documentation and IDE support

## 🙏 Acknowledgements

- This project was created as part of the 42 school curriculum
- Built with TensorFlow/Keras for deep learning capabilities
- Uses PlantCV for advanced plant image processing
- Leverages albumentations for state-of-the-art data augmentation

## 📄 License

This project is part of the 42 school curriculum and follows the school's guidelines for academic projects.
