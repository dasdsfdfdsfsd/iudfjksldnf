# RL for the evolution of cooperation with DP

### Requirements
1. Python 2.7
2. numpy
3. networkx

### Usage 

#### Part 1: Experiment1: to test the adaptivity in different structure networks and different proportion of cooperators
  
 - Training the model 
 
      ```Shell
     python main.py -n Homogeneous -r RL-DP -g Normal -fc 0.3,0.5,0.7 --experiment 1
     ```
 - For different networks, change 'Homogeneous' to 'Random' or 'ScaleFree' in the shell above to your desired one 
 - For different algorithm, change 'RL-DP' to 'BS', 'BN' or 'Redistribution'.
 - For different proportion of malicious agent, change 0.3,0.5,0.7 to any number range from 0 to 1. 
 - The last argument '--experiment 1' means the first experiment

#### Part 2: Experiment2: to test the stability to defend malicious agents, comparing to other schemes with and without malicious agents
```Shell
python main.py -n Homogeneous Random ScaleFree -r RL-DP -g Normal Malicious -fc 0.5 --experiment 2
```

#### Part 3: Experiment3: our algorithm in different noise levels and proportions of malicious agents
```Shell
python main.py -n Homogeneous -r RL-DP -g Malicious -fc 0.5 --experiment 3
```
#### Part 4: Dynamic networks: update networks every 100 rounds.
