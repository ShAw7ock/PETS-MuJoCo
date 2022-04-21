import argparse


def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", required=True, type=str,
                        help="environment name: GYMMB_* or Magellan*")
    parser.add_argument("--task_name", default="standard", type=str, help="assert standard")
    parser.add_argument("--seed", default=12, type=int, help="random seed")
    parser.add_argument("--normalize_data", default=False, type=bool, help="whether to normalize the data for training")
    parser.add_argument("--n_training_threads", default=None, type=int, help="while not GPU, set CPU threads")
    parser.add_argument("--n_total_steps", default=50000, type=int,
                        help="total number of steps in real environment (including warm up)")
    parser.add_argument("--n_warm_up_steps", default=1000, type=int, help="number of steps to initialized the buffer")
    parser.add_argument("--render", default=False, type=bool, help="rendering the env")
    parser.add_argument("--use_cuda", default=True, type=bool, help="use GPU")
    parser.add_argument("--batch_size", default=1024, type=int, help="Training batch size for agent")
    parser.add_argument('--eval_freq', default=1000, type=int, help="evaluating policy frequency")
    parser.add_argument('--grad_clip', default=5, type=int, help="gradient clip")
    parser.add_argument('--save_freq', default=10000, type=int, help="parameters save frequency")

    args = parser.parse_args()
    return args


def pets_args(args):
    args.plan_horizon = 20
    args.n_particles = 20
    args.population_size = 50
    args.num_elites = 10
    args.cem_max_iter = 5

    return args


def dynamics_model(args):
    # Model Network Parameters
    args.model_ensemble_size = 4                         # number of models in the bootstrap ensemble
    args.model_n_units = 512                             # number of hidden units in each hidden layer (hidden layer size)
    args.model_n_layers = 4                              # number of hidden layers in the model (at least 2)
    args.model_activation = 'swish'                      # activation function (see models.py for options)
    args.model_lr = 1e-4
    args.model_weight_decay = 1e-4
    args.model_grad_clip = 5
    # Model Training Parameters
    args.model_expand_steps = 10
    args.model_batch_size = 256
    args.model_training_freq = 25
    args.model_training_n_batches = 120
    return args
