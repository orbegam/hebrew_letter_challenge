# Hebrew Letter Challenge
Code for Tola's hebrew letter challenge.

## Goal
Train a neural network to identify handwritten hebrew letter (Alef, Bet and Gimel).

## Contents
**Hebrew_Letters_Challenge_-_Orb.ipynb** - A colab notebook with all the training process

**final_model_00175.h5** - A Keras model file with the final model for the challenge

**predict_letter.py** - A python script that identifies a single 81x81 image

## Usage
Place **predict_letter.py** and **final_model_00175.h5** in the same folder.

run **python predict_letter.py image_path [--verbose]**
