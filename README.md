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

As shown in the model above. States 
𝑆
=
{
𝑆
1
,
𝑆
2
,
.
.
.
,
𝑆
𝐾
}
S={S 
1
​
 ,S 
2
​
 ,...,S 
K
​
 }, where 
𝑆
𝑘
S 
k
​
  represents a hidden state.


Conclusion:

