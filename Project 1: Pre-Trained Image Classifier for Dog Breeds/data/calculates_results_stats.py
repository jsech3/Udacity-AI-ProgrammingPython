# calculates_results_stats.py
# Jack Sechler - 6/3/2025

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using the results_dic.
    Returns a dictionary containing the results stats (counts & percentages).

    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a List:
                    index 0 = pet image label (string)
                    index 1 = classifier label (string)
                    index 2 = match (1/0)
                    index 3 = is-a-dog (1/0)
                    index 4 = classifier labels as dog (1/0)

    Returns:
     results_stats_dic - Dictionary containing the results statistics.
    """
    stats = {
        'n_images': len(results_dic),
        'n_dogs_img': 0,
        'n_notdogs_img': 0,
        'n_match': 0,
        'n_correct_dogs': 0,
        'n_correct_notdogs': 0,
        'n_correct_breed': 0
    }

    for key, value in results_dic.items():
        is_match = value[2]
        is_dog = value[3]
        classified_as_dog = value[4]

        stats['n_match'] += is_match
        stats['n_dogs_img'] += is_dog
        stats['n_notdogs_img'] += 1 - is_dog

        if is_dog and classified_as_dog:
            stats['n_correct_dogs'] += 1
        elif not is_dog and not classified_as_dog:
            stats['n_correct_notdogs'] += 1

        if is_dog and is_match:
            stats['n_correct_breed'] += 1

    # Percentages
    stats['pct_match'] = (stats['n_match'] / stats['n_images']) * 100.0
    stats['pct_correct_dogs'] = (stats['n_correct_dogs'] / stats['n_dogs_img']) * 100.0 if stats['n_dogs_img'] > 0 else 0.0
    stats['pct_correct_breed'] = (stats['n_correct_breed'] / stats['n_dogs_img']) * 100.0 if stats['n_dogs_img'] > 0 else 0.0
    stats['pct_correct_notdogs'] = (stats['n_correct_notdogs'] / stats['n_notdogs_img']) * 100.0 if stats['n_notdogs_img'] > 0 else 0.0

    return stats
