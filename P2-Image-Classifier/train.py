#!/usr/bin/env python3
"""
train.py - Image Classifier Training Script
Udacity AI Programming with Python Nanodegree - Part 2

This script trains a neural network on a flower image dataset using transfer learning
with pretrained CNN models (VGG, ResNet, etc.).

Basic usage:
    python train.py flowers
    
Advanced usage:
    python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 3 --gpu --save_dir checkpoints
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import json
from collections import OrderedDict
import time


def get_input_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a neural network on flower images')
    
    # Required argument
    parser.add_argument('data_dir', type=str, 
                        help='Path to the data directory (e.g., flowers)')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--arch', type=str, default='vgg16',
                        choices=['vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units in classifier (default: 512)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    
    return parser.parse_args()


def load_data(data_dir):
    """
    Load and preprocess the image datasets
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Define transforms for training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx


def build_model(arch, hidden_units, num_classes):
    """
    Build the model with pretrained features and custom classifier
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


def validation(model, valid_loader, criterion, device):
    """
    Validate the model and return accuracy and loss
    """
    model.eval()
    valid_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss/len(valid_loader), accuracy/len(valid_loader)


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    """
    Train the model and print progress
    """
    steps = 0
    running_loss = 0
    print_every = 10
    
    print(f"Training on {device}")
    print(f"Training for {epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss, accuracy = validation(model, valid_loader, criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")
                
                running_loss = 0
                model.train()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print("-" * 50)


def save_checkpoint(model, arch, hidden_units, epochs, learning_rate, class_to_idx, save_dir):
    """
    Save the model checkpoint
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


def main():
    """
    Main function to coordinate the training process
    """
    # Get command line arguments
    args = get_input_args()
    
    # Set device
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA GPU training enabled")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Apple Silicon MPS GPU training enabled")
        else:
            device = torch.device("cpu")
            print("GPU requested but not available, using CPU")
    else:
        device = torch.device("cpu")
        print("CPU training")
    
    # Load data
    print("Loading data...")
    train_loader, valid_loader, test_loader, class_to_idx = load_data(args.data_dir)
    num_classes = len(class_to_idx)
    print(f"Found {num_classes} classes")
    
    # Build model
    print(f"Building {args.arch} model...")
    model = build_model(args.arch, args.hidden_units, num_classes)
    model.to(device)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters
    if args.arch.startswith('vgg'):
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    else:  # ResNet
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)
    
    # Save checkpoint
    save_checkpoint(model, args.arch, args.hidden_units, args.epochs, 
                   args.learning_rate, class_to_idx, args.save_dir)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()