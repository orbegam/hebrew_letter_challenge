# Hebrew Letter Challenge
Code for Tola's hebrew letter challenge.

## Goal
Train a neural network to identify handwritten hebrew letter (Alef, Bet and Gimel).

## Contents
**Hebrew_Letters_Challenge_Orb.ipynb** - A colab notebook with all the training process

**final_model_00175.h5** - A Keras model file with the final model for the challenge

**predict_letter.py** - A python script that identifies a single 81x81 image

**requirements.txt** - A requirements file for pip

## Usage
Place **predict_letter.py** and **final_model_00175.h5** in the same folder.

Make sure all prerequisites are installed using:

`pip install -r requirements.txt`

Run the prediction using:

`python predict_letter.py image_path [--verbose]`
