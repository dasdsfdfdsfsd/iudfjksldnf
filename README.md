# Multi-Agent

### Requirements
1. Python 2.7
2. numpy
3. networkx

### Usage 

#### Part 1: Experiment1: comparision between other methods and our algorithm without malicious agents
  
 - Training the model 
 
      ```Shell
     python main.py -n Homogeneous -r RL-DP -g Normal -fc 0.3,0.5,0.7 --experiment 1
     ```
 - For different networks, change 'Homogeneous' to 'Random' or 'ScaleFree' in the shell above to your desired one 
 - For different algorithm, change 'RL-DP' to 'BS', 'BN' or 'Redistribution'.
 - For different proportion of malicious agent, change 0.3,0.5,0.7 to any number range from 0 to 1. 
 - The last argument '--experiment 1' means the first experiment

#### Part 2: Experiment2: comparision between Normal(without malicious) and Malicious game 
```Shell
python main.py -n Homogeneous Random ScaleFree -r RL-DP -g Normal Malicious -fc 0.5 --experiment 2
```

#### Part 3: Experiment3: our algorithm in different noise level and proportion of malicious 
```Shell
python main.py -n Homogeneous -r RL-DP -g Malicious -fc 0.5 --experiment 3
```
