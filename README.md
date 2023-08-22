# HINDSIGHT STATES for Stable Baselines3 

This is the implementation of the algorithm Hindsight States (HiS) for Stable Baselines3. Hindsight States is a new algorithm, that aims to make Reinforcement Learning more efficient, by splitting the state into two parts ('real' and 'virtual'), and concurrently simulating multiple instances of the virtual part. The virtual parts are then relabelled 'in Hindsight'. For more details check out the paper linked below.

For an introduction to SB3 check out the [main repo](https://github.com/DLR-RM/stable-baselines3).


## Example 1: Fetch Robotics Environments

<img src="docs/\_static/img/his_fetch.png"  width="70%"/>

Here is an example of how to use HiS with a Fetch Robotics environment. It uses the Gymnasium-Robotics-HYSR package, that implements the Fetch environments with parallel virtual objects.

To reproduce the results of the paper, increase the number of timesteps. We recommend creating a new virtual environment to install the required Python packages.

Main prerequisites:
[Gymnasium-Robotics-HYSR](https://github.com/simonguist/Gymnasium-Robotics-HYSR) and [mujoco](https://pypi.org/project/mujoco/) 2.2.1 (or older).


```python
import gym
from stable_baselines3 import SAC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

log_path = "/tmp/logs"

# Apply HER and HiS on top of each other. To apply only HER or only HiS, set one of these flags to False
apply_HER = True
apply_HiS = True

def make_env_HySR():
    env = gym.make('FetchPushHysr1.0-v2')
    env = Monitor(env)  # record stats such as returns
    return env

def make_env_default():
    env = gym.make('FetchPush-v2', render_mode='human')
    env = Monitor(env)  # record stats such as returns
    return env

def make_logger():
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    return new_logger


# Training
env_HySR = DummyVecEnv([make_env_HySR])
logger_HySR=make_logger()
model = SAC("MultiInputPolicy", env_HySR, replay_buffer_class=HerReplayBuffer,
            # Use hyperparameters from rl-baselines3-zoo
            gamma = 0.95,
            tau = 0.05,
            ent_coef = "auto",
            learning_rate = 0.001,
            batch_size = 2048,
            gradient_steps = 1,
            train_freq = 1,
            buffer_size = 5000000,
            learning_starts = 1000,
            policy_kwargs={
                "net_arch": [512] * 3,
                "n_critics": 2
            },
            replay_buffer_kwargs=dict(
                hindsight_state_selection_strategy = "achieved_goal",  # options = ["random", "reward", "advantage", "achieved_goal"]
                hindsight_state_selection_strategy_horizon = "episode", # options = ["step", "future", "episode"]
                HSM_shape = 100,
                HSM_min_criterion = 0.02,
                HSM_n_traj_freq = 2000,
                max_episode_length= 50,
                n_sampled_hindsight_states = 3,
                n_sampled_goal = 4,
                HSM_goal_env = True,
                apply_HER = apply_HER,
                online_sampling = True,
                apply_HSM = apply_HiS,
                logger=logger_HySR,
            ),
            tensorboard_log = log_path,
            verbose=1)
model.set_logger(logger_HySR)
model.learn(total_timesteps=2000*50)    # 2000 episodes
env_HySR.close()

# Evaluation
env = DummyVecEnv([make_env_default])
obs = env.reset()
n_eps = 0
n_succ = 0
for i in range(100*50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        n_eps += 1
        if reward>-0.5:
            n_succ += 1
        obs = env.reset()
print("success rate: ", n_succ / n_eps)
env.close()
```

## Example 2: Table Tennis Environment

<img src="docs/\_static/img/his_tt.png"  width="50%"/>

Main prerequisites:
[PAM robot software](https://intelligent-soft-robots.github.io/pam_documentation/) and [learning_table_tennis_from_scratch](https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch)

Detailed instructions coming soon.




## Citing the Project

SB3 repository:

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

Hindsight States:

```bibtex
@article{Guist-RSS-23,
  author  = {Guist, Simon and Schneider, Jan and Dittrich, Alexander and Berenz, Vincent and Sch{\"o}lkopf, Bernhard and B{\"u}chler, Dieter},
  title   = {Hindsight States: Blending Sim and Real Task Elements for Efficient Reinforcement Learning},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year    = {2023},
  url     = {https://www.roboticsproceedings.org/rss19/p038.pdf}
}
```