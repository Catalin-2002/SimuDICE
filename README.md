
# SimuDICE

In offline reinforcement learning, deriving an effective policy from a pre-collected set of experiences is challenging due to the distribution mismatch between the *target policy* and the *behavioral policy* used to collect the data, as well as the limited sample size. Model-based reinforcement learning improves sample efficiency by generating simulated experiences using a learned dynamic model of the environment. However, these synthetic experiences often suffer from the same distribution mismatch. 

To address these challenges, we introduce **SimuDICE**, a framework that iteratively refines the initial policy derived from offline data using synthetically generated experiences from the world model. **SimuDICE** enhances the quality of these simulated experiences by adjusting the sampling probabilities of state-action pairs based on **stationary DIstribution Correction Estimation (DICE)** and the estimated confidence in the model's predictions. This approach guides policy improvement by balancing experiences similar to those frequently encountered with ones that have a distribution mismatch.

Our experiments show that **SimuDICE** achieves performance comparable to existing algorithms while requiring fewer pre-collected experiences and planning steps, and it remains robust across varying data collection policies.

**If you use SimuDICE in your work, please consider citing the original paper:**

> Catalin E. Brita, Stephan Bongers, and Frans A. Oliehoek, *SimuDICE: Offline Policy Optimization Through World Model Updates and DICE Estimation*. In *Proceedings of BNAIC/BENELEARN 2024*. [Link to paper](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/10/SimuDICE-Offline-Policy-Optimization-Through-World-Model-Updates-and-DICE-Estimation.pdf)

---

## Prerequisites

Follow these steps to set up the environment:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```
2. **Activate the virtual environment**:
   - **Linux/MacOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start Guide

### Step 1: Generate a Dataset
Create a dataset with the desired environment (e.g., `Taxi`) by running:
```bash
python3 dataset_generator.py \
    --env Taxi \
    --epsilon 0.1 \
    --alpha 0.1 \
    --gamma 0.99 \
    --train_episodes 10000 \
    --play_episodes 510 \
    --max_environment_steps 100 \
    --save_trajectories \
    --debug
```

### Step 2: Run the Main Algorithm
Optimize the policy using SimuDICE with the generated dataset:
```bash
python3 main.py \
    --env Taxi \
    --data_path ./datasets/Taxi_0.1_10000_510_100_behavioral_data.pkl \
    --alpha 0.1 \
    --gamma 0.99 \
    --planning_steps 20 \
    --iterations 1 \
    --lambda_value 100 \
    --sampling_strategy 1 \
    --play_episodes 500 \
    --max_environment_steps 100 \
    --max_episodes 100
```

---

## Command Descriptions

- `--env`: Specifies the environment (e.g., `Taxi`, `FrozenLake`).
- `--epsilon`: Exploration parameter for the epsilon-greedy policy.
- `--alpha`: Learning rate for Q-learning updates.
- `--gamma`: Discount factor for future rewards.
- `--train_episodes`: Number of episodes to train the initial policy.
- `--play_episodes`: Number of evaluation episodes.
- `--max_environment_steps`: Maximum steps allowed per episode.
- `--save_trajectories`: Saves the generated trajectories if enabled.
- `--debug`: Outputs detailed logs for debugging.
- `--data_path`: Path to the pre-collected dataset.
- `--planning_steps`: Number of planning iterations in SimuDICE.
- `--iterations`: Number of policy refinement iterations.
- `--lambda_value`: Regularization parameter for sampling probabilities.
- `--sampling_strategy`: Strategy for sampling synthetic experiences.

---

## Features

- **World Model Learning**: SimuDICE builds a simple model of the environment from offline datasets.
- **Distribution Correction**: Utilizes DICE estimations to adjust sampling probabilities and reduce distribution mismatch.
- **Policy Iteration**: Refines policies through synthetic experiences guided by confidence and distribution corrections.

---

## Citation

If you find this project useful, consider citing the associated paper:
```bibtex
@inproceedings{simudice2024,
  title={SimuDICE: Offline Policy Optimization Through World Model Updates and DICE Estimation},
  author={Brita, Catalin E. and Bongers, Stephan and Oliehoek, Frans A.},
  booktitle={BNAIC/BENELEARN 2024},
  year={2024},
  url={https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/10/SimuDICE-Offline-Policy-Optimization-Through-World-Model-Updates-and-DICE-Estimation.pdf}
}
```
