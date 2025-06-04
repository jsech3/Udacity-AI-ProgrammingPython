# adjust_results4_isadog.py
# Jack Sechler - 6/3/2025

def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if the classifier correctly
    classified images as 'a dog' or 'not a dog' using a file with dog names.

    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a List:
                    index 0 = pet image label (string)
                    index 1 = classifier label (string)
                    index 2 = 1/0 (1 = match, 0 = no match)
      dogfile - Text file with one dog name per line (string)

    Returns:
      None - results_dic is a mutable data type so no return needed.
    """
    # Read in dog names from file and store in a set
    dog_names = set()
    with open(dogfile, 'r') as f:
        for line in f:
            dog_names.add(line.strip().lower())

    # Adjust results_dic with flags for is-a-dog for both pet and classifier labels
    for key in results_dic:
        pet_label = results_dic[key][0]
        classifier_label = results_dic[key][1]

        # Check if pet label is a dog
        pet_is_dog = 1 if pet_label in dog_names else 0

        # Check if classifier label contains any known dog name
        classifier_is_dog = 0
        for name in dog_names:
            if name in classifier_label:
                classifier_is_dog = 1
                break

        # Append flags to results list
        results_dic[key].extend([pet_is_dog, classifier_is_dog])
