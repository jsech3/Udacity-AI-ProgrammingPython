# print_results.py
# Jack Sechler - 6/3/2025

def print_results(results_dic, results_stats, model, 
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (using input parameters).

    Parameters:
      results_dic - Dictionary with image filename as key and a List as value
      results_stats - Dictionary of results statistics
      model - CNN model architecture used
      print_incorrect_dogs - True prints incorrectly classified dog images
      print_incorrect_breed - True prints incorrectly classified dog breeds

    Returns:
           None - simply printing results.
    """
    print("\n*** Results Summary for CNN Model Architecture:", model.upper(), "***")
    print("Number of Images:", results_stats['n_images'])
    print("Number of Dog Images:", results_stats['n_dogs_img'])
    print("Number of Not-a-Dog Images:", results_stats['n_notdogs_img'])

    print("\nPercentage Stats:")
    print("% Match:", round(results_stats['pct_match'], 1))
    print("% Correct Dogs:", round(results_stats['pct_correct_dogs'], 1))
    print("% Correct Breed:", round(results_stats['pct_correct_breed'], 1))
    print("% Correct Not-a-Dog:", round(results_stats['pct_correct_notdogs'], 1))

    # Incorrect dog classifications
    if print_incorrect_dogs:
        print("\nIncorrect Dog/Not-a-Dog Assignments:")
        for filename, values in results_dic.items():
            is_dog = values[3]
            classified_as_dog = values[4]
            if is_dog != classified_as_dog:
                print(f"Misclassified: {filename} | True: {values[0]} | Classifier: {values[1]}")

    # Incorrect breed classifications
    if print_incorrect_breed:
        print("\nIncorrect Dog Breed Assignments:")
        for filename, values in results_dic.items():
            is_dog = values[3]
            classified_as_dog = values[4]
            match = values[2]
            if is_dog == 1 and classified_as_dog == 1 and match == 0:
                print(f"Wrong Breed: {filename} | True: {values[0]} | Classifier: {values[1]}")

