'''
Wrapper class for the model. It handles the training, the evaluation and meta-information.
'''

import gym
import numpy as np
import wandb
import wandb.sdk
from copy import deepcopy
from typing import Tuple
from stable_baselines3.ppo import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv


class ModelWrapper():
    def __init__(self, 
                 config_model : dict, 
                 meta_model : dict, 
                 meta_wandb : dict,
                 vertex : np.ndarray) -> None:
        self.config_model = config_model
        self.meta_model = meta_model
        self.meta_wandb = meta_wandb
        self.params_values = vertex

    # FIXME: can be removed?
    @property
    def training_env(self) -> gym.Env:
        return self._training_env
    
    @staticmethod
    def build_wrapper(config_model : dict, 
                      meta_model : dict, 
                      meta_wandb : dict, 
                      config_sampler : dict,
                      vertex : np.ndarray) -> Tuple['ModelWrapper', wandb.sdk.wandb_run.Run]:
        '''
        Create a new instance of ModelWrapper with the given parameters and with a model
        that has been trained with the given vertex parameters.

        Parameters
        ----------
        config_model : dict
            Configuration of the model;
        meta_model : dict
            Meta-configuration of the model;
        meta_wandb : dict
            Meta-configuration of WandB;
        config_sampler : dict
            Configuration of the sampler;
        vertex : np.ndarray
            Vertex parameters;

        Returns
        -------
        ModelWrapper
            A new instance of ModelWrapper with the given parameters.
        '''
        
        # Extract the parameters to estimate
        params = config_sampler['params']
        # Create a dictionary with the parameters and their values
        params_dict = {k: v for k, v in zip(params, vertex)}
        # Initialize the wrapper
        wrapper_new_vertex = ModelWrapper(config_model, meta_model, meta_wandb, vertex)
        wrapper_new_vertex._set_envs()
        # Pass the parameters to the training environment
        wrapper_new_vertex._set_physical_values(params_dict)
        wandb_callback, run = wrapper_new_vertex._build_model(params_dict)
        wrapper_new_vertex._model_learns(wandb_callback)
        return wrapper_new_vertex, run
    
    def _set_envs(self) -> None:
        '''
        Pass the already built training and validation environments to the wrapper.

        Parameters
        ----------
        training_env : gym.Env
            Training environment;
        validation_env : gym.Env
            Validation environment.
        '''
    
        self._training_env = make_vec_env(self.config_model['training_env'], 
                                          n_envs=self.meta_model['vectorized'], seed=42, vec_env_cls=SubprocVecEnv)

    def _set_physical_values(self, 
                            params_dict : dict) -> None:
        '''
        Set the masses of the training environment.

        Parameters
        ----------
        masses : np.ndarray
            Masses of the training environment.
        '''

        self._training_env.env_method('set_vals', **params_dict)

    def _build_model(self, params_dict) -> None:
        '''
        Instantiate a model with the given configuration.

        Parameters
        ----------
        params_dict : dict
            Dictionary with the parameters to estimate.
        '''

        run = wandb.init(
            project=self.meta_wandb['project'],
            config=self.config_model,
            sync_tensorboard=True
            )

        wandb_callback = WandbCallback(
            gradient_save_freq=self.meta_wandb['gradient_save_freq'],
            model_save_path=f'models/{run.id}',
            verbose=1,
            log='all'
        )
        
        run.log({**params_dict})

        ppo_config = deepcopy(self.config_model)
        # The environments must be removed to do dictionary unpacking
        del ppo_config['training_env']
        del ppo_config['validation_env']
        self.model = PPO(**ppo_config, env=self._training_env, tensorboard_log=f"runs/{run.id}")
        return wandb_callback, run

    def _model_learns(self, wandb_callback) -> None:
        '''
        Train a model with the given configuration.
        '''

        self.model.learn(
            total_timesteps=self.meta_model['total_timesteps'],
            log_interval=self.meta_wandb['log_interval'],
            callback=wandb_callback
            )

    # FIXME: can be removed?
    def load_model(self, 
                   model_path : str) -> None:
        '''
        Load a pretrained model from the given path.

        Parameters
        ----------
        model_path : str
            Path to the pretrained model.
        '''

        self.model = PPO.load(model_path)

    def evaluate_policy(self) -> Tuple[float, float, float]:
        '''
        Evaluate the policy of the model in the validation environment.

        Returns
        -------
        float
            Average reward;
        float
            Standard deviation of the rewards;
        float
            Average reward minus the standard deviation.
        '''

        rewards = []
        env = gym.make(self.config_model['validation_env'])
        
        for episode in range(10):
            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Predict the next action using the policy
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        env.close()

        # Calculate average and standard deviation of rewards
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        return avg_reward, std_reward, avg_reward - std_reward