# 🐶 Dog Breed Classifier – Udacity AI Programming with Python Project

I built this project as part of the Udacity AI Programming with Python Nanodegree. It uses pre-trained CNN models to classify pet images as dogs or not-dogs, and predicts the breed if it’s a dog. I tested three models — ResNet, AlexNet, and VGG — to compare their accuracy and runtime.

## 📁 Project Structure

```
Udacity-AI-ProgrammingPython/
├── data/
│   ├── pet_images/
│   ├── uploaded_images/
│   ├── check_images.py
│   ├── classifier.py
│   ├── print_results.py
│   ├── get_input_args.py
│   ├── calculates_results_stats.py
│   ├── adjust_results4_isadog.py
│   ├── dognames.txt
│   ├── resnet_pet-images.txt
│   ├── alexnet_pet-images.txt
│   ├── vgg_pet-images.txt
│   ├── resnet_uploaded-images.txt
│   ├── alexnet_uploaded-images.txt
│   └── vgg_uploaded-images.txt
├── README.md
```

## ⚙️ Setup Instructions

```bash
git clone https://github.com/jsech3/Udacity-AI-ProgrammingPython.git
cd Udacity-AI-ProgrammingPython
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # if available
```

## 🚀 Run the Classifier

```bash
python data/check_images.py --dir data/pet_images/ --arch resnet --dogfile data/dognames.txt
```

Swap `resnet` with `alexnet` or `vgg` to test other models.

To test all models at once:

```bash
sh data/run_models_batch.sh
```

To evaluate custom uploaded images:

```bash
sh data/run_models_batch_uploaded.sh
```

## 📊 Results Summary

| Architecture | % Match | % Correct Dogs | % Correct Breed | % Correct Not-a-Dog |
| ------------ | ------- | -------------- | --------------- | ------------------- |
| ResNet       | 82.5%   | 100.0%         | 90.0%           | 90.0%               |
| AlexNet      | 75.0%   | 100.0%         | 80.0%           | 90.0%               |
| VGG          | 87.5%   | 100.0%         | 93.3%           | 100.0%              |

## 🧪 Uploaded Image Findings

I tested four of my own images in the `uploaded_images/` folder:

* `Dog_01.jpg` – Doberman Pinscher
* `Dog_02.jpg` – Flipped Doberman
* `Alligator_01.jpg` – Not a dog
* `Sandwich_01.jpg` – Not a dog

All models correctly detected both Dobermans as dogs. However, none of the models got the breed right (even though the prediction said "Doberman Pinscher"). This suggests a mismatch between the expected label and the classifier string format.

The alligator was correctly flagged as "not a dog" by all three models. The sandwich was misclassified by both ResNet and VGG as a hot dog, while AlexNet called it a "french loaf" — technically still wrong but at least not labeled as a dog. So AlexNet had the highest accuracy on non-dog images (100% correct).

| Model   | Dog Accuracy | Breed Accuracy | Not-a-Dog Accuracy |
| ------- | ------------ | -------------- | ------------------ |
| ResNet  | 100%         | 0%             | 50%                |
| AlexNet | 100%         | 0%             | 100%               |
| VGG     | 100%         | 0%             | 50%                |

## 📚 Key Scripts

* `check_images.py` – Main pipeline
* `classifier.py` – Loads & applies chosen CNN
* `print_results.py` – Outputs stats
* `get_input_args.py` – Handles CLI input
* `calculates_results_stats.py` – Calculates metrics
* `adjust_results4_isadog.py` – Checks if labels are dog breeds

## 📅 Sample Output

```
*** Results Summary for CNN Model Architecture: RESNET ***
Number of Images: 40
Number of Dog Images: 30
Number of Not-a-Dog Images: 10

% Match: 82.5
% Correct Dogs: 100.0
% Correct Breed: 90.0
% Correct Not-a-Dog: 90.0
```

## 🚸️‍Author

**Jack Sechler**
[GitHub Repo](https://github.com/jsech3/Udacity-AI-ProgrammingPython)

---

## ✅ Notes

* All code and outputs are located in the `data/` folder.
* This README lives in the [Project 1: Pre-Trained Image Classifier for Dog Breeds](https://github.com/jsech3/Udacity-AI-ProgrammingPython/tree/main/Project%201%3A%20Pre-Trained%20Image%20Classifier%20for%20Dog%20Breeds) folder (same level as `data/`)
