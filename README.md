# Potato Disease Classifier using CNN

This project is a convolutional neural network (CNN) based classifier to detect three classes of potato leaves:
- Early Blight
- Late Blight
- Healthy

The model was built using TensorFlow and trained on a dataset of potato leaf images. It achieved a **93% accuracy on the test set** after just 10 epochs of training.

## Model Architecture
The model uses a CNN architecture with the following layers:
- Convolutional Layers
- Max Pooling Layers
- Dense Layers
- Dropout Layers (for regularization)

The model is trained using the Adam optimizer and categorical cross-entropy loss.

## Dataset
The dataset used contains images of potato leaves classified into three categories:
- Early Blight
- Late Blight
- Healthy

### Data Augmentation
To improve the model's performance and generalization, data augmentation techniques such as rotation, zoom, and horizontal flipping were applied to the training set.

## Results
- **Test Accuracy**: 93%
- **Training Time**: 10 epochs

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Saivarun2611/Potato_disease_Classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd potato-disease-classifier
    ```

3. Install the required dependencies
   
4. Run the script (optional if you want to retrain the model)
    

## Usage

You can use the trained model to predict the class of a new potato leaf image:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained TensorFlow model
model = load_model('potato_disease_classifier')

# Load and preprocess the image
img = image.load_img('path_to_image', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
classes = ['Early Blight', 'Late Blight', 'Healthy']
print(f'Predicted Class: {classes[np.argmax(prediction)]}')
