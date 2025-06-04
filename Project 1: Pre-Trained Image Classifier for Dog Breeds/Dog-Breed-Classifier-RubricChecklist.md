# 🔢 Dog Breed Classifier: Udacity Project Rubric Checklist

This document provides a clear mapping of how I met each requirement in the Udacity AI Programming with Python rubric.

---

## ⏱️ Timing Code

**Requirement:** Call time functions before the start of main code and after the main logic has finished.

* **Where I Did This:** `data/check_images.py`
* **Confirmed:** I used `start_time = time.time()` at the beginning and `end_time = time.time()` at the end to measure the execution duration.

---

## 🔹 Command Line Arguments

**Requirement:**

* Enable `--dir` with default `pet_images/`

* Enable `--arch` with default `vgg`

* Enable `--dogfile` with default `dognames.txt`

* **Where I Did This:** `data/get_input_args.py`

* **How I Use It:** Through `data/run_models_batch.sh` and `data/run_models_batch_uploaded.sh`

---

## 📃 Pet Image Labels

**Requirement:** Return a dictionary in the correct format with proper labels.

* **Where I Did This:** `data/get_pet_labels.py`
* **How I Verified It:**

  * Seen in `vgg_pet-images.txt` and `resnet_uploaded-images.txt`
  * Keys are filenames; values are lowercase, stripped labels
* **Confirmation:** 40 key-value pairs for pet images, 4 for uploaded image test

---

## 📷 Classifying Images

**Requirement:**

* Use correct path for classification

* Format labels

* Store results with match verification

* **Where I Did This:** `data/classify_images.py`

* **How I Call the Classifier:** `classifier(images_dir + filename, model)`

* **How I Process Labels:** I used `lower().strip()`

* **How I Store Results:** I used 1 for matches, 0 for mismatches in `results_dic`

---

## 🐕 Classifying Labels as Dogs

**Requirement:**

* Verify dog vs. not-dog classification

* Cross-reference both labels with dog names

* **Where I Did This:** `data/adjust_results4_isadog.py`

* **Support File:** `data/dognames.txt`

* **Proof:** Output in result `.txt` files shows `PetLabelDog` and `ClassLabelDog` columns

---

## 📊 Results

**Requirement:**

* Provide accurate results for 3 models

* Include key percentage statistics

* **Where I Calculate Stats:** `data/calculates_results_stats.py`

* **Where I Print Them:** `data/print_results.py`

* **Output Files I Created:**

  * `resnet_pet-images.txt`
  * `alexnet_pet-images.txt`
  * `vgg_pet-images.txt`
  * `resnet_uploaded-images.txt`
  * `alexnet_uploaded-images.txt`
  * `vgg_uploaded-images.txt`

---

## 🔧 Bonus: Uploaded Image Classification Test

* **Where My Images Are:** `data/uploaded_images/`

  * `Dog_01.jpg` (Doberman)
  * `Dog_02.jpg` (Flipped Doberman)
  * `Alligator_01.jpg` (Animal)
  * `Sandwich_01.jpg` (Object)

* **Batch Script I Used:** `data/run_models_batch_uploaded.sh`

* **Purpose:** To confirm model logic using non-pet\_images inputs

* **How I Evaluated:** I reviewed the results from each model's `.txt` file

---

## 📁 Final Notes

* I fulfilled all rubric requirements and clearly mapped my implementation.
* My code is modular, well-commented, and cleanly organized in the `data/` folder.
* I went beyond the base project by classifying new images and reviewing model outputs.