# Bayes Final Project

### Team: Adam's Family

Mitch Verhelle, Claire Fenton, Richard Huang

### Setup
- Put training data from kaggle into train, leave names as is: "input_2023_w[01..18].csv", "output_2023_w[01..18].csv"
- In the main folder, outside of any sub-folder, put "test_input.csv" and "test.csv" from Kaggle.
- pip install requirements (Python=3.13.7)


#### Project

This project implements Bayesian models for predicting NFL player movement using tracking data from the 2023 season. The goal is to predict the next-step position of players given their current kinematic state (position, speed, acceleration, direction) as well as predict whether or not the play was successful (i.e. the receiver caught the ball). The data and foundation for this project is from the NFL Big Data Bowl 2026 hosted on Kaggle. 

#### Overview

We developed multiple Bayesian approaches to model player movement, extending a deterministic kinematic model with probabilistic uncertainty quantification. The models use PyMC for Bayesian inference and provide posterior predictive distributions over future player positions.

#### Data

The NFL Big Data Bowl is an annual challenge hosted by the NFL that aims to examine how large amounts of football play data could generate real, potentially actionable, insights into various play metrics or strategies. This year, the data provided by the NFL included real-time data on player positions and movements, with the competition aiming for predicting player movements after the snap (based on their resulting X, Y coordinate on the field).

Alongside this, we also engineered a "successful_play" flag when the targeted receiver ended up within a meter's distance of the ball. This was used in predicting successful plays from previous player movements.


#### Models Implemented

1. **Bayesian Kinematic Model** (`bayes_kinematic.py`)
   - Extends a deterministic kinematic model (physics-based movement equations) with Bayesian uncertainty
   - Models noise in x and y (sigma_x, sigma_y) with HalfNormal priors (good for modeling standard deviations)
   - Uses MCMC (NUTS) for posterior sampling (NUTS is like MCMC but more efficiently picks where it samples)
<img width="963" height="372" alt="output2" src="https://github.com/user-attachments/assets/94e2954c-48d1-4d19-b5b1-35b01bcb2681" />
<img width="1175" height="497" alt="output3" src="https://github.com/user-attachments/assets/2d402e55-e2ea-49e5-9918-a8e05bb7d43a" />
   - Provides posterior predictive distributions for next-step positions
<img width="1160" height="499" alt="output" src="https://github.com/user-attachments/assets/06d1b775-c4d4-4b71-87e3-5a057e2c5ddb" />


![output](https://github.com/user-attachments/assets/38a71e83-3c11-4a0d-92f1-c4fa8182d08f)


2. **Hierarchical Bayesian Kinematic Model** (`bayes_kinematic_hierarchical.py`)
   - Adds hierarchical structure to account for player and position-specific differences
   - Global noise scales with position-specific deviations
   - Player-specific noise scales nested within positions
   - Captures that different player types (WR, RB, CB, etc.) may have different movement patterns and predictability
  
3. **Bayesian Logistic Regression Model** (`PyMC`)
   - Incorporates seven features/beta coefficients based on player/ball movement
   - Draws from Bernoulli likelihood distribution for binary outcome
   - Uses MCMC (NUTS) for posterior sampling
   - Predicts probability of play being successful (within 0.5 yards of target)

#### Methodology

1. **Data Processing**: Convert raw tracking data into step-level datasets with current kinematic features and next-step targets
2. **Model Fitting**: Use PyMC to perform Bayesian inference, learning posterior distributions over model parameters
3. **Prediction**: Generate posterior predictive samples for future positions, providing both point estimates (posterior means) and uncertainty quantification
4. **Evaluation**: Compare models on held-out test data and visualize predictions with uncertainty

#### Results

We developed a method to visualize both the kinematic components and the successful play results, seen in the demos provided in the repository.

