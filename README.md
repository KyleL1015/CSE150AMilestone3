# CSE150AMilestone3

Model: Milestone3.ipynb

Sources: 
https://hmmlearn.readthedocs.io/en/latest/tutorial.html
chatgpt.com
https://www.kaggle.com/datasets/ananthr1/weather-prediction

Overview: This project finds the most likely weather state given certain evidence such as minimum temperature, maximum temperature, wind speed, and precipitation using a Multinomial Hidden Markov Model. The "Multinomial" part of the the Multinomial HMM is because we are using this on discrete data values. By separating all of our evidence into bins such as "0-5 mm of precipitation = light rain" we can turn all of our continuous data into discrete values. 

PEAS: In terms of PEAS, the the reak world and its weather patterns. Specifically since the data is from Seattle it would include all the attributes in Seattle that the weather could contribute to. The performance measure of our model is is how accurately it predicts the weather (of past, present, and future states) given the observations mentioned before like temperature or precipitation. This model's actuators are the most likely output of weather conditions and the possible future states based on calculated probabilities. The sensors are simply the historical data that's being inputted into our model turned into discrete categories. 

Agent Type: We first preprocessed our data to get it ready for computation. We checked if rows had NULL values or blank columns and removed those with missing data. Since we needed categorical data in order to use Multinomial HMM, we turned all the continuous values of the HMM into discrete categories. For example, anything greater than 5 mm of precipitation is categorized as heavy rain. Out agrent is a passive model-based learning agent that uses probabilistic reasoning to reach conclusions. We train it on our initial dataset and generate values for our emission and transition matrices but it does not improve further unless retrained. Using these matrices and probabilistic reasoning algorithms such as Viterbis, we are able to generate likely weather sequences.

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

Emission Probability Calculation for Multiple Attributes:

For each observation at time $\( t \)$, which is a vector $\( O_t = (o_{t1}, o_{t2}, o_{t3}, o_{t4}) \)$, where $\( o_{t1} \)$ represents precipitation, $\( o_{t2} \)$ represents maximum temperature, $\( o_{t3} \)$ represents minimum temperature, and $\(o_{t4} \)$ represents the wind speed, the joint emission probability given a hidden state $\( S_i \)$ is computed as:

$\[
P(O_t \mid S_i) = \prod_{m=1}^{M} P(o_{tm} \mid S_i)
\]$

Where:
- $\( P(o_{tm} \mid S_i) \)$ is the probability of observing the $\( m \)$-th feature $\( o_{tm} \)$ (e.g., the temperature or wind speed) given hidden state $\( S_i \)$.
- This works because if you imagine each observation as a separate node (precipitation vs. wind), they are conditionally independent when conditioned on $\( S_n \)$ due to the fork condition of d-separation.
- $\( M \)$ is the total number of attributes (in this case, $\( M = 4 \)$).


Conclusion:

