# ml-from-scratch-python

A collection of machine learning algorithms implemented **from scratch** in Python, starting from the ground up to build a deep understanding of their inner workings.

The project is split into two stages:

* **Pure Python** (no external libraries): regression, perceptron and one-hot encoding. Only **matplotlib** is used, for visualization.
* **NumPy-based**: the neural network and CNN, where matrix operations are written with NumPy but forward pass, backpropagation and training loops are still implemented by hand (no TensorFlow, PyTorch or scikit-learn).

---

## ✅ Implemented

### 📊 Linear Regression

* [x] **Univariate** and **Multivariate** Linear Regression
* [x] **Full Batch Gradient Descent**: updates weights and bias using the gradient over the whole dataset
* [x] **Stochastic Gradient Descent**: updates on one randomly chosen sample at a time, using a hand-written **Linear Congruential Generator (LCG)** instead of the `random` module
* Uses **Mean Squared Error** loss
* Pure Python implementation
* Visualizations using **matplotlib** (1D line fit and 2D surface/plane fit, plus loss curves)

### 📉 Ridge Regression

* [x] **Univariate** and **Multivariate** Ridge Regression
* [x] **Full Batch Gradient Descent**
* MSE loss with an **L2 regularization** penalty (`reg`) added to the weight update to reduce overfitting
* Pure Python implementation
* Visualizations using **matplotlib**

### 🤖 Perceptron (Binary Classification)

* [x] **Univariate** and **Multivariate** Perceptron
* Uses **0-1 loss** and updates weights with the standard Perceptron rule
* Pure Python implementation
* Visualizations using **matplotlib** for both 1D and 2D examples

### 🔢 One-Hot Encoding

* [x] Converts categorical labels into one-hot vectors and returns the vocabulary for decoding predictions
* Pure Python implementation (`one-hot encoding.py`)
* The same function is reused in the Neural Network and CNN notebooks

### 🧠 Neural Network (4 layers)

* [x] Fully connected network: **3 hidden layers (ReLU) + softmax output layer**
* **Cross-entropy loss**, hand-written **backpropagation**, full-batch gradient descent
* **He initialization** for ReLU layers and Xavier-style initialization for the softmax layer
* Trained and evaluated on the **Iris** dataset (`Iris.csv`, included) with a 75/25 train/test split, reaching **100% accuracy**, plus a `Predict` helper that prints each prediction against the true label
* Also reached **83% accuracy** on the Kaggle [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) competition
* Uses **NumPy** and **pandas** (pandas only to load the CSV)

### 🖼️ Convolutional Neural Network

* [x] **Convolution (3×3 filters, ReLU) → 2×2 max pooling → flatten → dense (ReLU) → softmax**
* Convolution, max pooling and their backward passes are written by hand with explicit loops over patches
* **Cross-entropy loss** with **mini-batch gradient descent** and shuffling each epoch
* Trained on **MNIST** (IDX files included in `Convolutional Neural Network/Dataset/`)
* The example run trains on the first 1,000 training images (~93% training accuracy) and evaluates on the full 10,000-image test set (~85% test accuracy)
* Includes matplotlib visualizations of sample digits and misclassified examples
* Uses **NumPy**, **idx2numpy** (to read the MNIST files) and **matplotlib**

---

## 🚀 How to Run in Jupyter Notebook

1. Make sure you have Python installed, then install the dependencies:

   ```bash
   pip install notebook matplotlib numpy pandas idx2numpy
   ```

   (The pure Python notebooks only need `notebook` and `matplotlib`. `numpy`, `pandas` and `idx2numpy` are needed for the Neural Network and CNN.)

2. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open the notebook you want to explore:

   | Algorithm | Notebook |
   |---|---|
   | Linear Regression (Full Batch GD) | `Linear Regression/Full Batch GD/univariate.ipynb`<br>`Linear Regression/Full Batch GD/multivariate.ipynb` |
   | Linear Regression (Stochastic GD) | `Linear Regression/Stochastic GD/univariate.ipynb`<br>`Linear Regression/Stochastic GD/multivariate.ipynb` |
   | Ridge Regression (Full Batch GD) | `Ridge Regression/Full Batch GD/univariate.ipynb`<br>`Ridge Regression/Full Batch GD/multivariate.ipynb` |
   | Perceptron | `Perceptron/univariate.ipynb`<br>`Perceptron/multivariate.ipynb` |
   | One-Hot Encoding | `One-Hot Encoding/one-hot encoding.py` (run with `python`) |
   | Neural Network | `Neural Network/4_layer_nn.ipynb` |
   | Convolutional Neural Network | `Convolutional Neural Network/3_layer_cnn.ipynb` |

4. Run the cells in the notebook to see the implementation and results.

> **Note:** Start Jupyter from inside the notebook's own folder (or keep the folder structure intact), since the Neural Network notebook reads `Iris.csv` and the CNN notebook reads `Dataset/...` using relative paths.

---