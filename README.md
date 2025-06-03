# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: Puttepu Venkata Balarama Vamsi

*INTERN ID*: CT04DN135

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

## 🧠 Task 3: CNN for Image Classification on MNIST Dataset – Detailed Description

In the third task of the CODTECH Machine Learning Internship, we built a **Convolutional Neural Network (CNN)** for image classification using the **MNIST dataset**. This task marked our first foray into **deep learning**, where we explored how CNNs process and classify images, particularly grayscale images of handwritten digits (0–9).

### 📁 Dataset Overview

The **MNIST** dataset is a benchmark dataset in machine learning, consisting of 70,000 grayscale images of handwritten digits, each of size 28x28 pixels. The dataset is split into 60,000 training images and 10,000 testing images. Each image is labeled with the digit it represents, making this a multi-class classification problem with 10 classes (digits 0–9).

### 🔄 Data Preprocessing

Before feeding the data into the CNN, we performed preprocessing steps:
- **Normalization**: Pixel values were scaled from the range [0, 255] to [0, 1] to improve training efficiency.
- **Reshaping**: Each image was reshaped from a 2D array (28x28) into a 3D array (28x28x1) to match the input format expected by Keras CNN layers.

### 🧠 CNN Architecture

We built the CNN model using TensorFlow and Keras. The architecture consisted of the following layers:
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 1**: 2x2 pooling
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 2**: 2x2 pooling
- **Flatten Layer**: Converts 2D feature maps to 1D feature vector
- **Dense Layer**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation for multi-class output

The model was compiled using the **Adam optimizer** and **sparse categorical crossentropy** as the loss function, with accuracy as the evaluation metric.

### 📈 Model Training and Evaluation

The model was trained on the training set for 5 epochs with 10% of the data used for validation. During training, the model’s performance was monitored on both the training and validation sets.

After training, we evaluated the model on the 10,000 test images. The model achieved over **98% accuracy**, demonstrating its effectiveness in recognizing handwritten digits.

### 📊 Performance Metrics

We evaluated the model using:
- **Accuracy Score**: Percentage of correctly predicted digits
- **Classification Report**: Precision, recall, F1-score for each digit class
- **Confusion Matrix**: Visual representation of model performance across all digit classes

The confusion matrix revealed where the model made the most errors, which typically occurred between digits with similar shapes like ‘4’ and ‘9’ or ‘3’ and ‘8’.

### 🧪 Predictions

We also used the model to predict new images from the test dataset and compared predictions with actual labels, which showed that the CNN was able to generalize well even on unseen data.

### 🗂 Project Output

All code was written in a Jupyter Notebook, compatible with **Google Colab** for convenience and ease of execution. The project is organized and commented for clarity.

### ✅ Conclusion

This task demonstrated the power of convolutional neural networks in solving computer vision problems. The MNIST digit classifier serves as a foundational deep learning project, preparing us for more complex image classification tasks. We learned about CNN layers, activation functions, training strategies, and evaluation methods. This knowledge will be essential for future work involving deep learning and image processing.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/b8482fd8-9ba1-4477-bb35-8baadd656775)

![Image](https://github.com/user-attachments/assets/688d449e-ae93-458e-868e-093ab500002b)
