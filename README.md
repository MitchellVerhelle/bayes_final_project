# Bayes Final Project

### Team: Adam's Family

Mitch Verhelle, Claire Fenton, Richard Huang

### Setup
- Put training data from kaggle into train, leave names as is: "input_2023_w[01..18].csv", "output_2023_w[01..18].csv"
- In the main folder, outside of any sub-folder, put "test_input.csv" and "test.csv" from Kaggle.
- pip install requirements (Python=3.13.7) (Recommended: uv sync using uv.lock and pyproject.toml)


#### Project

https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview

This project implements Bayesian models for predicting NFL player movement using tracking data from the 2023 season. The goal is to predict the next-step position of players given their current kinematic state (position, speed, acceleration, direction) as well as predict whether or not the play was successful (i.e. the receiver caught the ball). The data and foundation for this project is from the NFL Big Data Bowl 2026 hosted on Kaggle. 

#### Overview

We developed multiple Bayesian approaches to model player movement, extending a deterministic kinematic model with probabilistic uncertainty quantification. The models use PyMC for Bayesian inference and provide posterior predictive distributions over future player positions.

#### Data

The NFL Big Data Bowl is an annual challenge hosted by the NFL that aims to examine how large amounts of football play data could generate real, potentially actionable, insights into various play metrics or strategies. This year, the data provided by the NFL included real-time data on player positions and movements, with the competition aiming for predicting player movements after the snap (based on their resulting X, Y coordinate on the field).

Alongside this, we also engineered a "successful_play" flag when the targeted receiver ended up within a meter's distance of the ball. This was used in predicting successful plays from previous player movements.

**Files**

train/

input_2023_w[01-18].csv

The input data contains tracking data before the pass is thrown

- `game_id`: Game identifier, unique (numeric)
- `play_id`: Play identifier, not unique across games (numeric)
- `player_to_predict`: whether or not the x/y prediction for this player will be scored (bool)
- `nfl_id`: Player identification number, unique across players (numeric)
- `frame_id`: Frame identifier for each play/type, starting at 1 for each game_id/play_id/file type (input or output) (numeric)
- `play_direction`: Direction that the offense is moving (left or right)
- `absolute_yardline_number`: Distance from end zone for possession team (numeric)
- `player_name`: player name (text)
- `player_height`: player height (ft-in)
- `player_weight`: player weight (lbs)
- `player_birth_date`: birth date (yyyy-mm-dd)
- `player_position`: the player's position (the specific role on the field that they typically play)
- `player_side`: team player is on (Offense or Defense)
- `player_role`: role player has on play (Defensive Coverage, Targeted Receiver, Passer or Other Route Runner)
- `x`: Player position along the long axis of the field, generally within 0 - 120 yards. (numeric)
- `y`: Player position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)
- `s`: Speed in yards/second (numeric)
- `a`: Acceleration in yards/second^2 (numeric)
- `o`: orientation of player (deg)
- `dir`: angle of player motion (deg)
- `num_frames_output`: Number of frames to predict in output data for the given game_id/play_id/nfl_id. (numeric)
- `ball_land_x`: Ball landing position position along the long axis of the field, generally within 0 - 120 yards. (numeric)
- `ball_land_y`: Ball landing position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)

output_2023_w[01-18].csv

The output data contains tracking data after the pass is thrown.

- `game_id`: Game identifier, unique (numeric)
- `play_id`: Play identifier, not unique across games (numeric)
- `nfl_id`: Player identification number, unique across players. (numeric)
- `frame_id`: Frame identifier for each play/type, starting at 1 for each game_id/play_id/ file type (input or output). The maximum value for a given game_id, play_id and nfl_id will be the same as the num_frames_output value from the corresponding input file. (numeric)
- `x`: Player position along the long axis of the field, generally within 0-120 yards. (TARGET TO PREDICT)
- `y`: Player position along the short axis of the field, generally within 0 - 53.3 yards. (TARGET TO PREDICT)

https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data

#### Models Implemented

1. **Bayesian Kinematic Model** (`bayes_kinematic.py`)

Position updates:

\[
x_{t+\Delta t} = x_t 
    + s_t \cos(\theta_t)\,\Delta t 
    + \tfrac{1}{2} a_t \cos(\theta_t)\,\Delta t^2
\]

\[
y_{t+\Delta t} = y_t 
    + s_t \sin(\theta_t)\,\Delta t 
    + \tfrac{1}{2} a_t \sin(\theta_t)\,\Delta t^2
\]

Observed next-step positions are modeled as Gaussian noise around the deterministic means:

\[
x_{\text{next}} \sim \mathcal{N}(\mu_x, \sigma_x)
\]

\[
y_{\text{next}} \sim \mathcal{N}(\mu_y, \sigma_y)
\]

Priors on noise:

\[
\sigma_x \sim \text{HalfNormal}(1.0)
\]

\[
\sigma_y \sim \text{HalfNormal}(1.0)
\]

This means we expect 0-1+ yards of noise, but very small amount of noise, on a weak prior.

Posterior Inference. Samples from the joint posterior:

\[
p(\sigma_x, \sigma_y \mid x_{\text{next}}, y_{\text{next}}, \mu_x, \mu_y)
\]

This is how we pick sigma_x and sigma_y for the next step.

Posterior Predictive Distribution. Future movement samples are drawn as:

\[
x^\*(t+\Delta t) \sim \mathcal{N}(\mu_x, \sigma_x)
\]

\[
y^\*(t+\Delta t) \sim \mathcal{N}(\mu_y, \sigma_y)
\]

So we use the drawn \(\sigma_x\) and \(\sigma_y\) to pull our next predicted frame. Then we roll out more than 1 frame of prediction to get a stronger idea of where the player will move next.

   - Extends a deterministic kinematic model (physics-based movement equations) with Bayesian uncertainty
   - Models noise in x and y (sigma_x, sigma_y) with HalfNormal priors (good for modeling standard deviations)
   - Uses MCMC (NUTS) for posterior sampling (NUTS is like MCMC but more efficiently picks where it samples)
<img width="963" height="372" alt="output2" src="https://github.com/user-attachments/assets/94e2954c-48d1-4d19-b5b1-35b01bcb2681" />
<img width="1175" height="497" alt="output3" src="https://github.com/user-attachments/assets/2d402e55-e2ea-49e5-9918-a8e05bb7d43a" />
   - Provides posterior predictive distributions for next-step positions
<img width="1160" height="499" alt="output" src="https://github.com/user-attachments/assets/06d1b775-c4d4-4b71-87e3-5a057e2c5ddb" />


2. **Hierarchical Bayesian Kinematic Model** (`bayes_kinematic_hierarchical.py`)
   - Adds hierarchical structure to account for player and position-specific differences
   - Global noise scales with position-specific deviations
   - Player-specific noise scales nested within positions
   - Captures that different player types (WR, RB, CB, etc.) may have different movement patterns and predictability
  
     Hierarchies followed the following levels:

     1. Global (Basic kinematic)
     2. Position Specific (WR, RB, etc.)
     3. Player Specific
    
     Starting from 2, we can apply a unique set of position-specific parameters for each unique position:

     ```
     σ_x_pos[i] ~ Normal(σ_x_global, 0.3)  # Position noise centered at global
      σ_y_pos[i] ~ Normal(σ_y_global, 0.3)
      
      bias_x_pos[i] ~ Normal(0, 0.2)  # Position-specific systematic biases
      bias_y_pos[i] ~ Normal(0, 0.2)
     ```
  
With this, we developed a heatmap visualization method to visualize how players could move in a given play. The heatmap visualization methodology is as follows:

      1. Posterior sampling: For each of the players at time step t after the snap, generate a number of posterior samples (compute deterministic mean from kinematic (μ_x, μ_y), 
      sample from Normal(μ_x, σ_x) and Normal(μ_y, σ_y) using learned noise scales)
      
      2. Sample validation: Filters samples to valid field coordinates (0 ≤ x ≤ 120, 0 ≤ y ≤ 53.3) to ensure that the potential movements are at least on the football field (no teleporting for our players...)
      
      3. KDE density estimation: Creates a grid on the football field and computes probability density at each grid point based on the above samples
      
      4. Visual tweaking: emphasize high-density regions and overlays heatmaps at t+1, t+2, and t+3 on field.


![alt text](https://github.com/MitchellVerhelle/bayes_final_project/blob/main/demo_frames/frame0000037.png "Post Snap Frame")


   see demo.html for more!
   
  
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

Example with `BayesianKinematicModel`
![output](https://github.com/user-attachments/assets/38a71e83-3c11-4a0d-92f1-c4fa8182d08f)

Example with `BayesianPlaySuccessModel` (in mitch_branch)

https://github.com/user-attachments/assets/7159380c-a971-485b-abae-a77b06273e71

https://github.com/user-attachments/assets/94016daa-61c6-43c1-a390-71c0b0ff2d10

Betas
- `tackle_range`
<img width="690" height="390" alt="tackle_range vs p_success" src="https://github.com/user-attachments/assets/5cc2f870-a246-4dd1-b8ce-2db2fdbfb08a" />
<img width="661" height="152" alt="beta tackle_range" src="https://github.com/user-attachments/assets/2e40b5ad-2540-4d1a-8a5d-857b511e3656" />

- `dist_to_nearest_defender`
<img width="689" height="390" alt="dist_to_nearest_defender" src="https://github.com/user-attachments/assets/443d86e7-d87e-4b46-b2c3-4a86afcf6843" />
<img width="661" height="152" alt="beta dist_to_nearest_defender" src="https://github.com/user-attachments/assets/b7a20d2e-d0ce-42fa-af49-3173ed6af548" />

- `dist_to_target_land`
<img width="690" height="390" alt="dist_to_target_land vs p_success" src="https://github.com/user-attachments/assets/a451b9d9-9ddb-42d1-a6f4-fbeffe57a446" />
<img width="661" height="153" alt="beta dist_to_target_land" src="https://github.com/user-attachments/assets/5c53886b-0199-4341-b3dd-57899041571c" />


