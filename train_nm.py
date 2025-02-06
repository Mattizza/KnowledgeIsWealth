import yaml
import argparse
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

    config = yaml_dict['hyperparameters_ppo']
    meta_wandb = yaml_dict['meta_wandb']
    meta_model = yaml_dict['meta_model']

    ############
    # OPTIMIZE #
    ############

    if args.method == 'nnm':
        solver = NelderMead(args=yaml_sampler_dict, config_model=config, meta_wandb=meta_wandb,
                            meta_model=meta_model, config_sampler=yaml_sampler_dict)
    elif args.method == 'inm':
        solver = IncompleteNelderMead(args=yaml_sampler_dict, config_model=config, meta_wandb=meta_wandb,
                            meta_model=meta_model, config_sampler=yaml_sampler_dict)
    elif args.method == 'anm':
        solver = ApproximateNelderMead(args=yaml_sampler_dict, config_model=config, meta_wandb=meta_wandb,
                            meta_model=meta_model, config_sampler=yaml_sampler_dict)
    else:
        raise ValueError("Method not recognized. Choose from: 'nnm', 'inm', 'anm'")
    solver.search()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    
    parser.add_argument('--path_config', '-p', type=str, default=None, help='Path to the configuration for the SAC model')
    parser.add_argument('--path_sampler_config', '-ps', type=str, default=None, help='Path to the configuration of the sampler')
    parser.add_argument('--method', '-m', type=str, default = None, help='Method to use for the optimization')
    args = parser.parse_args()

    main(args)