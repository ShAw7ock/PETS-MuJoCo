Probabilistic Ensemble and Trajectory Sampling (PETS)
============================

Implementation for Model-based Reinforcement Learning algorithms [PETS](https://proceedings.neurips.cc/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf).

## NOTE
* Wrapped MuJoCo environments bound the action spaces in the scale of `[-1, 1]`.
* Run the PETS experiments with: `python ./main.py --env_name ENV_NAME` which can be seen in: `./envs/gymmb/__init__.py`.
* Modify the Hyper-parameters in: `./components/arguments.py`
* Ensemble Models can be seen in: `./components/dynamics.py`
* Cross Entropy Method (CEM) sampling can be seen in: `./components/cem.py`
* There is not absolutely ***POLICY*** network to map states to actions and agent can be seen in: `./algo/pets.py`

## Requirements
* Python >= 3.6.0
* PyTorch == 1.7.0 (optional)
* [MUJOCO 200](https://roboti.us/)
* [mujoco-py](https://github.com/openai/mujoco-py)
* OpenAI Gym

# Acknowledgement
This code is referenced by [nnaisense/MAGE](https://github.com/nnaisense/MAGE). <br>
