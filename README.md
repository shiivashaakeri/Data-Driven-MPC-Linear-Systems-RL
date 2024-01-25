# Data-driven MPC for Linear Systems using Reinforcement Learning
## Abstract
 This project implements a novel scheme to address the optimal control problem for unknown linear systems in a data-driven manner. The method doesn't require prior knowledge of the system. It only utilizes past input-output trajectories to implicitly describe the system features and realize the prediction on the basis of behavioral systems theory. By adopting reinforcement learning to update the terminal cost function online, the stability and performance of the system are ensured. This approach bypasses the need for system identification, which can be difficult in practice, and instead learns from online measurements.

## Problem Formulation
$$ J^* _L  = \min\limits _ {\alpha (t), \sigma (t), \bar{u}(t), \bar{y}(t)} \sum _{k=0}^{L-1} l(\bar{u} _k(t), \bar{y} _k(t)) + \lambda _{\alpha} \bar{\epsilon} ||{\alpha (t)}|| _2^2 + \lambda _{\sigma}  ||{\sigma (t)}|| _2^2 + V _L(y _L) $$
## Algorithm
1. Capture I/O trajectory
2. Initializations: 
	Set RL learning rate, prediction horizon, cost matrices, regularization coefficients, reference state, choose polynomial basis vector, initialize weight, choose small weight convergence threshold
3. Generate Hankels
4. Solve the optimization problem, get optimal predictive control sequence
5. Update weight until convergence
6. Apply the first value of the optimal predictive control sequence to the system and store the new I/O to the trajectory
7. Set t= t+1, go back to 3.

## Citation 
[1] Z. Sun, Q. Wang, J. Pan and Y. Xia, "Data-Driven MPC for Linear Systems using Reinforcement Learning," 2021 China Automation Congress (CAC), Beijing, China, 2021, pp. 394-399, doi: 10.1109/CAC53003.2021.9728233. keywords: {Linear systems;Costs;Reinforcement learning;Predictive models;Prediction algorithms;Stability analysis;Trajectory;Model predictive control (MPC);reinforcement learning (RL);data-driven method},

[2] Berberich, Julian & Köhler, Johannes & Muller, Matthias & Allgöwer, Frank. (2020). Data-Driven Model Predictive Control With Stability and Robustness Guarantees. IEEE Transactions on Automatic Control. PP. 1-1. 10.1109/TAC.2020.3000182. 
