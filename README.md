# SoccerFoulProject

## Overview
SoccerFoulProject aims to aid football referees by providing a second opinion on the nature of specific actions and the associated penalties using AI. The model analyzes replays of actions from different angles to predict the nature of the action and any applicable penalty.

## Features
- **Action Detection**: Identifies various actions such as "Tackling," "Standing tackling," "High leg," "Holding," "Pushing," "Elbowing," "Challenge," and "Dive."
- **Penalty Prediction**: Determines the severity of the action and suggests the appropriate penalty:
  - No offence: 0
  - Offence + No card
  - Offence + Yellow card
  - Offence + Red card

## Model Architecture
The model architecture is inspired by the VARS paper. Check it out [here](https://arxiv.org/abs/2304.04617).

### Video Encoding
Each replay clip is passed to a video encoder to compute features. We tested three video encoders:

- **r3d_18**: A 3D ResNet model that extends the 2D convolutional filters to 3D, allowing the model to capture temporal features across multiple frames. This makes it well-suited for video classification tasks.
  
- **mc3_18**: An MC3 model that modifies the 3D ResNet by replacing some of the 3D convolutions with mixed 3D and 2D convolutions. This model aims to balance between capturing spatial and temporal features efficiently.

- **r2plus1d_18**: This model decomposes 3D convolutions into separate spatial and temporal convolutions (2+1D). This decomposition helps in better learning of spatial and temporal features independently, improving the model's performance on video data.

### Feature Aggregation
Computed features from the different clips are aggregated either by averaging or taking the maximum values.

### Classification Heads
The final aggregated features are passed to two classification heads:
- **Action Classification**: Predicts the action class.
- **Penalty Classification**: Predicts the offence/severity label.

## Acknowledgements
- [VARS paper](https://arxiv.org/abs/2304.04617)
