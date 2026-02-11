# 🖼️ CIFAR-10 Image Classification using ANN & CNN  

## 📌 Project Overview  
This project applies **Deep Learning techniques** to classify images from the CIFAR-10 dataset using both:

- Artificial Neural Network (ANN)  
- Convolutional Neural Network (CNN)  

The goal is to compare performance and demonstrate why CNNs are more effective for image classification tasks.

---

## 📊 Dataset  
**CIFAR-10 Dataset**  
- 60,000 RGB images  
- 10 classes  
- Image size: 32x32  

### Classes:
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

Split:
- 50,000 Training Images  
- 10,000 Testing Images  

---

## 🛠️ Technologies Used  
- Python  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- TensorFlow / Keras  

---

## 🧠 Models Implemented  

### 🔹 Artificial Neural Network (ANN)
- Flatten Layer  
- Dense (512 neurons, ReLU)  
- Dense (400 neurons, ReLU)  
- Output Layer (Softmax – 10 classes)  

### 🔹 Convolutional Neural Network (CNN)
- 3 Convolutional Blocks  
- Conv2D Layers  
- Batch Normalization  
- MaxPooling  
- Dropout (Regularization)  
- Fully Connected Layers  
- Softmax Output Layer  

---

## ⚙️ Preprocessing  
- Labels reshaped  
- Pixel values normalized (0–1 scaling)  
- Train/Test split provided by dataset  

---

## 📈 Model Evaluation  
- Accuracy Score  
- Classification Report  
- Confusion Matrix  

The CNN model achieved higher accuracy and better generalization compared to ANN due to its ability to capture spatial features.

---

## 💾 Saved Model  
```
Images Classification.h5
```

---

## 🚀 Key Learning Outcomes  
- Understanding ANN vs CNN differences  
- Building Deep Learning pipelines  
- Image preprocessing  
- Regularization using Dropout & BatchNormalization  
- Multi-class classification  

---

## 👨‍💻 Author  
**Mohamed Shaban**  
AI Engineering Student  
Interested in Computer Vision & Healthcare AI  
