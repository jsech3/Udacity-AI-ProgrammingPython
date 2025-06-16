#!/usr/bin/env python3
"""
predict.py - Image Classifier Prediction Script
Udacity AI Programming with Python Nanodegree - Part 2

This script uses a trained neural network to predict the class of an input image.

Basic usage:
    python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth
    
Advanced usage:
    python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
from collections import OrderedDict


def get_input_args():
    """
    Parse command line arguments for prediction
    """
    parser = argparse.ArgumentParser(description='Predict flower class from an image')
    
    # Required arguments
    parser.add_argument('image_path', type=str,
                        help='Path to the input image')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to the model checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=1,
                        help='Return top K most likely classes (default: 1)')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """
    Load the trained model from checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model architecture and parameters
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    class_to_idx = checkpoint['class_to_idx']
    
    # Rebuild the model
    model = build_model(arch, hidden_units, len(class_to_idx))
    
    # Load the trained weights
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set class to index mapping
    model.class_to_idx = class_to_idx
    
    # Create index to class mapping for predictions
    model.idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return model


def build_model(arch, hidden_units, num_classes):
    """
    Rebuild the model architecture (same as in train.py)
    """
    # Load pretrained model
    if arch.startswith('vgg'):
        model = getattr(models, arch)(pretrained=True)
        input_size = model.classifier[0].in_features
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, hidden_units)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(hidden_units, num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        model.classifier = classifier
        
    elif arch.startswith('resnet'):
        model = getattr(models, arch)(pretrained=True)
        input_size = model.fc.in_features
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, hidden_units)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(hidden_units, num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        model.fc = classifier
    
    return model


def process_image(image_path):
    """
    Process image for prediction
    Scales, crops, and normalizes a PIL image for a PyTorch model
    """
    # Open the image
    pil_image = Image.open(image_path)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(pil_image)
    
    return image_tensor


def predict(image_path, model, device, top_k=1):
    """
    Predict the class (or classes) of an image using a trained deep learning model
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Process the image
    image_tensor = process_image(image_path)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        
    # Calculate probabilities
    probabilities = torch.exp(output)
    
    # Get top k probabilities and classes
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
    
    # Convert to lists
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Convert indices to class labels
    top_classes = [model.idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes


def load_category_names(category_names_path):
    """
    Load category names from JSON file
    """
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def display_prediction(image_path, top_probs, top_classes, cat_to_name=None, top_k=1):
    """
    Display prediction results
    """
    print(f"\nPrediction for image: {image_path}")
    print("-" * 50)
    
    if top_k == 1:
        # Single prediction
        class_name = cat_to_name[top_classes[0]] if cat_to_name else top_classes[0]
        print(f"Predicted class: {class_name}")
        print(f"Probability: {top_probs[0]:.4f}")
    else:
        # Top K predictions
        print(f"Top {top_k} predictions:")
        for i in range(len(top_probs)):
            class_name = cat_to_name[top_classes[i]] if cat_to_name else top_classes[i]
            print(f"{i+1:2d}. {class_name:25s} - Probability: {top_probs[i]:.4f}")
    
    print("-" * 50)


def main():
    """
    Main function to coordinate the prediction process
    """
    # Get command line arguments
    args = get_input_args()
    
    # Set device
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU for prediction")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon MPS GPU for prediction")
        else:
            device = torch.device("cpu")
            print("GPU requested but not available, using CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction")
    
    try:
        # Load the trained model
        print(f"Loading model from: {args.checkpoint_path}")
        model = load_checkpoint(args.checkpoint_path, device)
        print("Model loaded successfully")
        
        # Load category names if provided
        cat_to_name = None
        if args.category_names:
            cat_to_name = load_category_names(args.category_names)
            print(f"Category names loaded from: {args.category_names}")
        
        # Make prediction
        print(f"Processing image: {args.image_path}")
        top_probs, top_classes = predict(args.image_path, model, device, args.top_k)
        
        # Display results
        display_prediction(args.image_path, top_probs, top_classes, cat_to_name, args.top_k)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == '__main__':
    main()