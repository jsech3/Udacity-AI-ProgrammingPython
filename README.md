# 🐶 Dog Breed Classifier – Udacity AI Programming with Python Project

This project is part of the Udacity AI Programming with Python Nanodegree. It uses a pre-trained Convolutional Neural Network (CNN) to classify pet images as either dogs or not-dogs, and for dog images, it predicts the breed. The goal is to evaluate three CNN architectures—**ResNet**, **AlexNet**, and **VGG**—in terms of accuracy and runtime.

---

## 📁 Project Structure

```
Udacity-AI-ProgrammingPython/
├── data/
│   ├── pet_images/
│   ├── check_images.py
│   ├── classifier.py
│   ├── print_results.py
│   ├── get_input_args.py
│   ├── calculates_results_stats.py
│   ├── adjust_results4_isadog.py
│   ├── dognames.txt
│   ├── resnet_pet-images.txt
│   ├── alexnet_pet-images.txt
│   └── vgg_pet-images.txt
├── project-workspace-classify-uploaded-images/
├── project-workspace-printing-results/
├── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jsech3/Udacity-AI-ProgrammingPython.git
   cd Udacity-AI-ProgrammingPython
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements.txt  # If you create one
   ```

---

## 🚀 Run the Classifier

To classify pet images with a selected CNN model:

```bash
python data/check_images.py --dir data/pet_images/ --arch resnet --dogfile data/dognames.txt
```

Replace `resnet` with `alexnet` or `vgg` to test other models.

### ✉️ Batch Test All Models:

```bash
sh data/run_models_batch.sh
```

Generates:

* `resnet_pet-images.txt`
* `alexnet_pet-images.txt`
* `vgg_pet-images.txt`

---

## 📊 Results Summary

| Architecture | % Match | % Correct Dogs | % Correct Breed | % Correct Not-a-Dog |
| ------------ | ------- | -------------- | --------------- | ------------------- |
| ResNet       | 82.5%   | 100.0%         | 90.0%           | 90.0%               |
| AlexNet      | 75.0%   | 100.0%         | 80.0%           | 90.0%               |
| VGG          | 87.5%   | 100.0%         | 93.3%           | 100.0%              |

---

## 📚 Function Descriptions

* **`check_images.py`**: Orchestrates the classification pipeline.
* **`classifier.py`**: Loads a specified CNN model to classify images.
* **`print_results.py`**: Displays summary stats and errors.
* **`get_input_args.py`**: Handles CLI input.
* **`calculates_results_stats.py`**: Computes accuracy stats.
* **`adjust_results4_isadog.py`**: Verifies if labels are dog breeds.

---

## 📅 Sample Output

```
*** Results Summary for CNN Model Architecture: RESNET ***
Number of Images: 40
Number of Dog Images: 30
Number of Not-a-Dog Images: 10

Percentage Stats:
% Match: 82.5
% Correct Dogs: 100.0
% Correct Breed: 90.0
% Correct Not-a-Dog: 90.0
```

---

## 🚸️‍Author

**Jack Sechler**
[GitHub Repo](https://github.com/jsech3/Udacity-AI-ProgrammingPython)

---

## ✅ Notes

* All code and outputs are located in the `data/` folder.
* The workspace folders prefixed with `project-workspace-` were part of the Udacity course steps and can be safely ignored or deleted if you're submitting or showcasing the final result.
* The `README.md` should live in the **root folder** of the project (same level as `data/`).