<h2 align="center"><strong>Actor-Critic Model Predictive Control</strong></h2>
<h3 align="center">Differentiable Optimization meets Reinforcement Learning for Agile Flight</h3>

<p align="center">
  <img src="img/acmpc_structure.png" alt="ACMPC Architecture" width="550"><br>
  <a href="https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf"><strong>📄 Paper (PDF)</strong></a> •
  <a href="https://www.youtube.com/watch?v=_qekrF4Emzg"><strong>🎥 Video</strong></a>
</p>

> ### Abstract  
> A key open challenge in agile quadrotor flight is how to combine the flexibility and task-level generality of model-free reinforcement learning (RL) with the structure and online replanning capabilities of model predictive control (MPC), aiming to leverage their complementary strengths in dynamic and uncertain environments. This paper provides an answer by introducing a new framework called Actor-Critic Model Predictive Control. The key idea is to embed a differentiable MPC within an actor-critic RL framework. This integration allows for short- term predictive optimization of control actions through MPC, while leveraging RL for end-to-end learning and exploration over longer horizons. Through various ablation studies, conducted in the context of agile quadrotor racing, we expose the benefits of the proposed approach: it achieves better out-of-distribution behaviour, better robustness to changes in the quadrotor’s dy- namics and improved sample efficiency. Additionally, we conduct an empirical analysis using a quadrotor platform that reveals a relationship between the critic’s learned value function and the cost function of the differentiable MPC, providing a deeper understanding of the interplay between the critic’s value and the MPC cost functions. Finally, we validate our method in a drone racing task on different tracks, in both simulation and the real world. Our method achieves the same superhuman performance as state-of-the-art model-free RL, showcasing speeds of up to 21 m/s. We show that the proposed architecture can achieve real-time control performance, learn complex behaviors via trial and error, and retain the predictive properties of the MPC to better handle out-of-distribution behavior.

This repository offers an implementation of Actor-Critic Model Predictive Control (ACMPC) for agile quadrotor flight, built on top of Differentiable Model Predictive Control and integrated with Stable-Baselines3.

The contents are:
- A quadrotor dynamics model (`diff_mpc_drones/drone.py`), which implements the forward dynamics of a quadrotor model and its jacobians
- An MPC “environment” wrapper (`diff_mpc_drones/il_env.py`)
- Training modules that plug both MPC and MLP policies into Stable-Baselines3 policies (`training_modules/`)
- External dependencies:
  - `mpc.pytorch` – differentiable MPC solver
  - `stable-baselines3` – RL library (fork with small modifications for ACMPC)
  
> **Citation (BibTeX):**
> ```bibtex
> @article{romero2025acmpc,
>   author  = {Romero, Angel and Aljalbout, Elie and Song, Yunlong and Scaramuzza, Davide},
>   title   = {Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning for Agile Flight},
>   journal = {IEEE Transactions on Robotics},
>   year    = {2025},
> }
> ```
> ```


