# CSE150AMilestone3

Model: Milestone3.ipynb

Sources: 
https://hmmlearn.readthedocs.io/en/latest/tutorial.html
chatgpt.com
https://www.kaggle.com/datasets/ananthr1/weather-prediction

Overview: This project finds the most likely weather state given observed data such as minimum temperature, maximum temperature, wind speed, and precipitation using a Gaussian Hidden Markov Model. This HMM is referred to as 'Gaussian' because it assumes the observations follow a Normal/Gaussian distribution. We use a Gaussian Hidden Markov Model because our weather features are continuous numerical values rather than discrete values.

PEAS: In terms of PEAS, the environment consists of real-world weather patterns, specifically those of Seattle, where the dataset originates. The performance measure of our model is its accuracy in predicting weather states (past, present, and future) based on observed data such as temperature, wind speed, and precipitation. The higher its accuracy, the better its performance. The actuators in our model are the predicted probability distribution over possible weather states. The sensors are the historical weather data input into the model which consists of a sequence of continuous numerical observations.

Agent Type: We first preprocessed our data to get it ready for computation. We checked if rows had NULL values or blank columns and removed those with missing data. Our agent is a passive model-based learning agent that uses probabilistic reasoning to reach conclusions. Specifically, it uses a Gaussian Hidden Markov Model to predict weather states. The agent is trained on an initial dataset to estimate the emission and transition matrices which define the probabilistic relationships between hidden weather states and observed weather features. However, it does not actively update or improve its model in response to new data unless explicitly retrained. Using these learned probability distributions and algorithms such as the Viterbi algorithm, the agent infers the most likely sequence of weather states given a sequence of observations.


![alt text](weatherHMM.jpg)


As seen in our above diagram, this Hidden Markov Model (HMM) is defined by:
(Equations and LaTex done with help from Chatgpt)
- States $\( S = \{S_1, S_2, \dots, S_n\} \)$, where $\( S_n \)$ represents a hidden weather state.
- Observations $\( O = \{O_1, O_2, \dots, O_T\} \)$, where $\( O_t \)$ is a vector of multiple observed features at time step $\( t \)$. Each observation is made up of 
  $\[
  O_t = (\text{Precipitation}_t, \text{MaxTemperature}_t, \text{MinTemperature}_t, \text{Wind Speed}_t)
  \]$
- Transition probabilities $\( A = [A_{ij}] \)$, where $\( A_{ij} \)$ is the probability of transitioning from state $\( S_i \) to \( S_j \)$.
  $\[
  A_{ij} = P(S_t = S_j \mid S_{t-1} = S_i)
  \]$
- Emission probabilities $\( B = [B_{ik}] \)$, where $\( B_{ik} \)$ is the probability of observing $\( O_t \)$ given hidden state $\( S_i \)$.
  $\[
  B_{ik} = P(O_t = O_k \mid S_t = S_i)
  \]$
  Where $\( O_k \)$ is an observation vector $\( O_t = (\text{Precipitation}_t, \text{MaxTemperature}_t, \text{MinTemperature}_t, \text{Wind Speed}_t) \)$.

Emission Probability Calculation for Gaussian HMM:

For each observation at time $\( t \)$, which is a vector $\( O_t = (o_{t1}, o_{t2}, o_{t3}, o_{t4}) \)$, where $\( o_{t1} \)$ represents precipitation, $\( o_{t2} \)$ represents maximum temperature, $\( o_{t3} \)$ represents minimum temperature, and $\(o_{t4} \)$ represents the wind speed, the joint emission probability given a hidden state $\( S_i \)$ is computed as:

$\[
P(o_t \mid s_t = i) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_i|}} \exp\left(-\frac{1}{2}(o_t - \mu_i)^\top \Sigma_i^{-1}(o_t - \mu_i)\right)
\]$

Where:
- $\( P(o_{tm} \mid S_i) \)$ is the probability of observing the $\( m \)$-th feature $\( o_{tm} \)$ (e.g., the temperature or wind speed) given hidden state $\( S_i \)$.
- This works because if you imagine each observation as a separate node (precipitation vs. wind), they are conditionally independent when conditioned on $\( S_n \)$ due to the fork condition of d-separation.
- $\( M \)$ is the total number of attributes (in this case, $\( M = 4 \)$).


Conclusion:

