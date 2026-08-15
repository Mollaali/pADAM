import yaml
from argparse import ArgumentParser
from scripts import var_params

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--offset', type=int, help='Offset value')
    parser.add_argument('--typePDE', type=str, help='Name of the PDE')
    parser.add_argument('--Nobs', type=int, help='Number of observations')
    parser.add_argument('--TypeProblem', type=str, help='Type of problem')
    parser.add_argument('--seed', type=int, help='Seed value')
    parser.add_argument('--device_index', type=int, help='Device index')

    options = parser.parse_args()
    config_path = options.config
    offset = options.offset
    typePDE = options.typePDE
    Nobs = options.Nobs
    TypeProblem = options.TypeProblem
    seed = options.seed
    device_index = options.device_index

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']

    if name == 'var_params':
        var_params(config, offset, typePDE, Nobs, TypeProblem, seed, device_index)
