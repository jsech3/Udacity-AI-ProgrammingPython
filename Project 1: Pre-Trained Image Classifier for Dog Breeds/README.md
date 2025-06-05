# 🐶 Dog Breed Classifier – Udacity AI Programming with Python Project

I built this project as part of the Udacity AI Programming with Python Nanodegree. It uses pre-trained CNN models to classify pet images as dogs or not-dogs, and predicts the breed if it's a dog. I tested three models ResNet, AlexNet, and VGG to compare their accuracy and runtime.

## 🎯 Project Objectives

1. **Correctly identify which pet images are of dogs** (even if the breed is misclassified) and which pet images aren't of dogs
2. **Correctly classify the breed of dog** for the images that are of dogs
3. **Determine which CNN model architecture** (ResNet, AlexNet, or VGG) "best" achieves objectives 1 and 2
4. **Consider the time resources required** to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each algorithm takes to run

## 📋 Requirements

- Python 3.6+
- PyTorch
- PIL/Pillow
- torchvision
- argparse

## 📁 Project Structure

```
Udacity-AI-ProgrammingPython/
├── data/
│   ├── pet_images/                    # Test images provided by Udacity
│   ├── uploaded_images/               # Custom test images
│   ├── check_images.py               # Main pipeline
│   ├── classifier.py                 # CNN model loader
│   ├── print_results.py              # Output statistics
│   ├── get_input_args.py             # CLI argument parser
│   ├── calculates_results_stats.py   # Metrics calculator
│   ├── adjust_results4_isadog.py     # Dog breed checker
│   ├── get_pet_labels.py             # Pet label extractor
│   ├── classify_images.py            # Image classifier
│   ├── dognames.txt                  # List of dog breed names
│   ├── resnet_pet-images.txt         # ResNet results
│   ├── alexnet_pet-images.txt        # AlexNet results
│   ├── vgg_pet-images.txt            # VGG results
│   ├── resnet_uploaded-images.txt    # ResNet custom results
│   ├── alexnet_uploaded-images.txt   # AlexNet custom results
│   └── vgg_uploaded-images.txt       # VGG custom results
├── README.md
```

## ⚙️ Setup Instructions

```bash
git clone https://github.com/jsech3/Udacity-AI-ProgrammingPython.git
cd Udacity-AI-ProgrammingPython
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision pillow
```

## 🚀 Run the Classifier

### Single Model Test
```bash
python data/check_images.py --dir data/pet_images/ --arch resnet --dogfile data/dognames.txt
```

Swap `resnet` with `alexnet` or `vgg` to test other models.

### Test All Models (Batch)
```bash
sh data/run_models_batch.sh
```

### Evaluate Custom Uploaded Images
```bash
sh data/run_models_batch_uploaded.sh
```

## 📊 Results Summary

### Performance on Udacity Pet Images Dataset

| Architecture | % Match | % Correct Dogs | % Correct Breed | % Correct Not-a-Dog | Runtime     |
|--------------|---------|----------------|-----------------|---------------------|-------------|
| ResNet       | 82.5%   | 100.0%         | 90.0%           | 90.0%               | 00:00:12    |
| AlexNet      | 75.0%   | 100.0%         | 80.0%           | 90.0%               | 00:00:07    |
| VGG          | 87.5%   | 100.0%         | 93.3%           | 100.0%              | 00:00:35    |

### Performance on Custom Uploaded Images

| Model   | Dog Accuracy | Breed Accuracy | Not-a-Dog Accuracy | Runtime     |
|---------|--------------|----------------|---------------------|-------------|
| ResNet  | 100%         | 0%             | 50%                 | 00:00:02    |
| AlexNet | 100%         | 0%             | 100%                | 00:00:01    |
| VGG     | 100%         | 0%             | 50%                 | 00:00:10    |

****⚠️Breed accuracy is 0% for all models*** on the uploaded images not because the classifier misidentified the breed, but because the filenames (`Dog_01.jpg`, `Dog_02.jpg`) were too generic. The pet label extracted was just `"dog"`, which doesn't match the more specific classifier output like `"doberman, doberman pinscher"`. This is expected in this project, unless filenames include the specific breed name (e.g., `doberman_01.jpg`).

## 🧪 Custom Image Testing Findings

I tested four of my own images in the `uploaded_images/` folder:

- **`Dog_01.jpg`** – Doberman Pinscher
- **`Dog_02.jpg`** – Flipped Doberman
- **`Alligator_01.jpg`** – Not a dog
- **`Sandwich_01.jpg`** – Not a dog

**Key Observations:**
- All models correctly detected both Dobermans as dogs
- None of the models got the breed right (label formatting mismatch)
- The alligator was correctly flagged as "not a dog" by all models
- The sandwich was misclassified by ResNet and VGG as a hot dog
- AlexNet correctly identified the sandwich as a "french loaf" (not a dog)

## 🧪 Questions regarding Uploaded Image Classification:

### 1. Did the three model architectures classify the breed of dog in Dog_01.jpg to be the same breed? If not, report the differences in the classifications.

**Answer:** Yes, all three models classified Dog_01.jpg as "doberman, doberman pinscher".

### 2. Did each of the three model architectures classify the breed of dog in Dog_01.jpg to be the same breed of dog as that model architecture classified Dog_02.jpg? If not, report the differences in the classifications.

**Answer:** Yes, all models classified both Dog_01.jpg and Dog_02.jpg as "doberman, doberman pinscher".

### 3. Did the three model architectures correctly classify Animal_Name_01.jpg and Object_Name_01.jpg to not be dogs? If not, report the misclassifications.

**Answer:** All models correctly classified Alligator_01.jpg as not a dog. For Sandwich_01.jpg: AlexNet correctly identified it as "french loaf" (not a dog), while ResNet and VGG misclassified it as "hotdog" (interpreted as a dog).

### 4. Based upon your answers for questions 1. - 3. above, select the model architecture that you feel did the best at classifying the four uploaded images. Describe why you selected that model architecture as the best on uploaded image classification.

**Answer:** AlexNet performed best because it achieved 100% accuracy on not-a-dog classification, correctly identifying the sandwich as "french loaf" while ResNet and VGG misclassified it as "hotdog".

## 📚 Key Scripts

| Script | Purpose |
|--------|---------|
| `check_images.py` | Main pipeline orchestrator |
| `classifier.py` | Loads & applies chosen CNN |
| `get_input_args.py` | Handles CLI argument parsing |
| `get_pet_labels.py` | Extracts labels from filenames |
| `classify_images.py` | Classifies images using CNN |
| `adjust_results4_isadog.py` | Checks if labels are dog breeds |
| `calculates_results_stats.py` | Calculates performance metrics |
| `print_results.py` | Outputs performance statistics |

## 📅 Sample Output

```
*** Results Summary for CNN Model Architecture: VGG ***
Number of Images: 40
Number of Dog Images: 30
Number of Not-a-Dog Images: 10

Percentage Stats:
% Match: 87.5
% Correct Dogs: 100.0
% Correct Breed: 93.3
% Correct Not-a-Dog: 100.0

** Total Elapsed Runtime: 0:00:35
```

## 🏆 Conclusion

**VGG had the best overall performance** with 87.5% match accuracy and 93.3% correct breed identification, making it the most reliable model for this task. However, AlexNet was fastest and correctly classified the custom object image as non-dog, showing its strength in real-world edge cases.

**Key Takeaways:**
- All models excel at distinguishing dogs from non-dogs
- VGG provides the most accurate breed classifications
- Custom image testing revealed challenges with label formatting
- AlexNet showed superior performance on custom non-dog images

## 👨‍💻 Author

**Jack Sechler**  
[GitHub Repository](https://github.com/jsech3/Udacity-AI-ProgrammingPython)

## 📄 License

This project is part of the Udacity AI Programming with Python Nanodegree program.
## ✅ Notes

* All code and outputs are located in the `data/` folder.
* This README lives in the [Project 1: Pre-Trained Image Classifier for Dog Breeds](https://github.com/jsech3/Udacity-AI-ProgrammingPython/tree/main/Project%201%3A%20Pre-Trained%20Image%20Classifier%20for%20Dog%20Breeds) folder (same level as `data/`)
