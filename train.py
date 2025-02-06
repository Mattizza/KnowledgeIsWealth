import gym
import wandb
import yaml
import argparse
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from extension.samplers import NelderMead, IncompleteNelderMead, ApproximateNelderMead

def main(args):

    #################
    # CONFIGURATION #
    #################

    # Load the configs
    with open(args.path_config, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    
    with open(args.path_sampler_config, 'r') as stream:
        yaml_sampler_dict = yaml.safe_load(stream)

    config = yaml_dict['hyperparameters_sac']
    meta_wandb = yaml_dict['meta_wandb']
    meta_model = yaml_dict['meta_model']

    config['training_env'] = args.training_env
    config['validation_env'] = args.validation_env
    config['vectorized'] = args.vectorized
    config['udr_lowhigh'] = meta_model['udr_lowhigh']

    # Build the environments
    training_env = gym.make(id=args.training_env)
    validation_env = gym.make(id=args.validation_env)    

    # Initialize a new run
    run = wandb.init(
        project=meta_wandb['project'],
        config=config,
        sync_tensorboard=True
        )

    del config['training_env']
    del config['validation_env']
    del config['vectorized']
    del config['udr_lowhigh']


    ############################
    # MONITOR AND MULTIPROCESS #
    ############################

    # Upload stats on WandB
    wandb_callback = WandbCallback(
            gradient_save_freq=meta_wandb['gradient_save_freq'],
            model_save_path=f'models/{run.id}',
            verbose=1,
            log='all'
        )

    # Handle vectorization, logging and UDR
    if args.vectorized is not None:
        print(f"Vectorized envs with {args.vectorized} cores")
        training_env = make_vec_env(args.training_env, n_envs=args.vectorized, seed=42, vec_env_cls=SubprocVecEnv)
        validation_env = make_vec_env(args.validation_env, n_envs=args.vectorized, seed=42, vec_env_cls=SubprocVecEnv)

        if 'udr_lowhigh' in meta_model.keys():
            print(f"UDR (percentages: {meta_model['udr_lowhigh']})")
            training_env.env_method('set_randomization', meta_model['udr_lowhigh'])
            validation_env.env_method('set_randomization', None)
            training_env.env_method('link_wandb_run', run)

    elif 'udr_lowhigh' in meta_model.keys():
        print("Vectorized envs are not used")
        print(f"UDR (percentages: {meta_model['udr_lowhigh']})")
        training_env.set_randomization(meta_model['udr_lowhigh'])
        validation_env.set_randomization(None)
        training_env.link_wandb_run(run)

    # Store the best model of a single run, tested every meta_model['eval_freq'] timesteps
    save_best_callback = EvalCallback(validation_env, 
                                best_model_save_path=f'./best_models/{run.id}/',
                                log_path=f'./best_models/{run.id}/',
                                eval_freq=meta_model['eval_freq'],
                                deterministic=False, render=False)

    ############
    # OPTIMIZE #
    ############
    solver = NelderMead(args=yaml_sampler_dict, config_model=config)
    all_solutions, all_fitnesses, best_solutions, init_points = solver.search()



    ############
    # LEARNING #
    ############

    # Create a model and learn
    model = SAC(**config, env=training_env, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=meta_model['total_timesteps'],
        log_interval=meta_wandb['log_interval'],
        callback=[wandb_callback, save_best_callback]
    )
    run.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()    
    parser.add_argument('--path_config', '-p', type=str, default=None, help='Path to the configuration for the SAC model')
    parser.add_argument('--path_sampler_config' '-ps', type=str, default=None, help='Path to the configuration of the sampler')
    parser.add_argument('--training_env', '-te', type=str, default='CustomHopper-source-v0', help='Environment over which to perform the training')
    parser.add_argument('--validation_env', '-ve', type=str, default='CustomHopper-source-v0', help='Environment over which to perform the validation')
    parser.add_argument('--vectorized', '-v', type=int, default = None, help='If given, set the number of processes for a vectorized environment')
    args = parser.parse_args()

    main(args)