# classify_images.py
# Jack Sechler - 6/3/2025

from classifier import classifier
import os

def classify_images(images_dir, results_dic, model):
    """
    Creates classifier labels with the classifier function, compares them to the
    pet image labels, and adds the classifier label and the comparison result
    to the results_dic.

    Parameters:
      images_dir - The path to the folder of images (string)
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
                    List. The list contains the following item:
                        index 0 = pet image label (string)
      model - CNN model architecture to use for classification (string)

    Returns:
      None - results_dic is a mutable data type so no return needed.
    """
    for filename in results_dic:
        # Get full path to image file
        image_path = os.path.join(images_dir, filename)

        # Get classifier label using specified model
        classifier_label = classifier(image_path, model)
        classifier_label = classifier_label.lower().strip()

        # Get pet label from results_dic
        pet_label = results_dic[filename][0]

        # Check if pet label is found in classifier label string
        match = 1 if pet_label in classifier_label else 0

        # Append classifier label and match result
        results_dic[filename].extend([classifier_label, match])
