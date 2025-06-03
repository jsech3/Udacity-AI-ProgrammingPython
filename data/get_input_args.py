#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                   
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following 3 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these 3 command line arguments.
    If the user fails to provide any arguments, default values are used.

    Returns:
        parse_args() - data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    # Command line argument 1: image directory
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='Path to folder of pet images')

    # Command line argument 2: CNN architecture
    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model architecture to use: vgg, alexnet, or resnet')

    # Command line argument 3: dog names file
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='Text file that contains dog names')

    return parser.parse_args()
