# 📘 Udacity AI Programming with Python Nanodegree

This repository contains both required projects for Udacity's AI Programming with Python Nanodegree.

---

## ✅ Project 1: Use a Pre-trained Image Classifier to Identify Dog Breeds

**Status:** Completed  
**Folder:** [Project 1: Pre-Trained Image Classifier for Dog Breeds](https://github.com/jsech3/Udacity-AI-ProgrammingPython/tree/main/Project%201%3A%20Pre-Trained%20Image%20Classifier%20for%20Dog%20Breeds)

### Overview
Developed a Python-based command-line tool that uses a pre-trained CNN to:
- Determine whether an image contains a dog, human, or neither
- If a dog (or human) is detected, identify the dog breed using a classifier architecture (AlexNet, VGG, or ResNet)

### Key Components
- `check_images.py`: CLI script to run the classifier and interpret results  
- Custom `classifier.py` module for transfer learning  
- Directory and argument parsing using `argparse`  
- Data loading and model evaluation using PyTorch  

---

## ✅ Project 2: Create Your Own Image Classifier

**Status:** Completed  
**Folder:** [Project 2: Image Classifier Development & Testing](https://github.com/jsech3/Udacity-AI-ProgrammingPython/tree/main/Project%202%3A%20Image%20Classifier%20Development%20%26%20Testing)

### Overview
Built a deep learning image classifier using PyTorch and transfer learning on the 102-category flower dataset.

### Key Components
- `train.py`: Trains a new model using a specified architecture (e.g., VGG16), with GPU support and checkpoint saving  
- `predict.py`: Loads a saved checkpoint and predicts the top K most likely flower categories from a given image  
- Transfer learning with pre-trained models and a custom classifier head  
- Configurable CLI inputs (e.g., hidden units, learning rate, number of epochs, GPU toggle)  
- Model validation and testing for performance evaluation  

---

> All work in this repository was completed by **Jack Sechler** and is hosted at [github.com/jsech3](https://github.com/jsech3).
