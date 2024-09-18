import argparse


def get_options():
    parser = argparse.ArgumentParser(description="Marco")

    # Environment
    parser.add_argument('--n', type=int, default=20, help='Problem size')
    parser.add_argument('--revisit_punishment', type=float, default=1.0, help='Revisit punishment')
    parser.add_argument('--local_opt_reward', type=float, default=0.0, help='Local opt reward')

    # Memory
    parser.add_argument('--memory_type', type=str, default='individual', choices=['shared', 'individual', 'op_based', 'none'], help='Memory type: shared, individual, operation-based, none')
    parser.add_argument('--memory_size', type=int, default=100000, help='Memory size')
    parser.add_argument('--k', type=int, default=20, help='k')
    parser.add_argument('--mem_aggr', type=str, default='linear', choices=['sum', 'linear', 'exp'], help='The aggregation method for the memory. Sum of all k values. Linear weighted sum. Exponential weighted sum.')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--n_layers', type=int, default=3, help='n_layers')
    parser.add_argument('--shared_weights', action='store_true', help='Use shared weights')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
    parser.add_argument('--normalization', type=str, default='layer', choices=['batch', 'instance', 'layer', 'rms'], help='Normalization')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'swiglu', 'relu'], help='Activation')
    parser.add_argument('--bias', action='store_true', help='Use bias')
    parser.add_argument('--tanh_clipping', type=float, default=10, help='Tanh clipping')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision. Only for evaluation')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='Betas')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--n_episodes', type=int, default=100, help='n_episodes')
    parser.add_argument('--episode_length', type=int, default=10, help='Episode length')
    parser.add_argument('--train_patience', type=int, default=3, help='Train patience')
    parser.add_argument('--max_train_steps', type=int, default=100, help='Max train steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.95, help='Gamma')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max grad norm')
    parser.add_argument('--rand_train_sizes', action='store_true', help='Random train sizes')

    # Test
    parser.add_argument('--max_test_steps', type=int, default=200, help='Max test steps')
    parser.add_argument('--topk', type=int, default=1, help='Executed actions in each inference step. topk=1 for default inference')
    parser.add_argument('--n_solutions', type=int, default=10, help='Number of solutions in population')
    parser.add_argument('--test_patience', type=int, default=1000, help='Test patience')

    # Others
    parser.add_argument('--model_load_enable', type=bool, default=False, help='Model load enable')
    parser.add_argument('--model_load_path', type=str, default=f'saved_models/', help='Model load path')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--save_model_freq', type=int, default=1, help='Save model freq')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose')

    args = parser.parse_args()
    # group all arguments in different dictionaries
    env_params = {
        'problem_size': args.n,
        'revisit_punishment': args.revisit_punishment,
        'local_opt_reward': args.local_opt_reward,
    }
    memory_params = {
        'memory_type': args.memory_type,
        'memory_size': args.memory_size,
        'k': args.k,
        'mem_aggr': args.mem_aggr,
    }
    model_params = {
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'shared_weights': args.shared_weights,
        'n_heads': args.n_heads,
        'normalization': args.normalization,
        'activation': args.activation,
        'bias': args.bias,
        'tanh_clipping': args.tanh_clipping,
        'dropout': args.dropout,
        'mixed_precision': args.mixed_precision,
    }
    optimizer_params = {
        'optimizer': {
            'lr': args.lr,
            'betas': args.betas,
            'weight_decay': args.weight_decay,
        },
    }
    trainer_params = {
        'epochs': args.epochs,
        'n_episodes': args.n_episodes,
        'episode_length': args.episode_length, # for calculating discounted rewards
        'train_patience': args.train_patience,  # Variable episode length
        'max_train_steps': args.max_train_steps,
        'batch_size': args.batch_size,
        'gamma': args.gamma,  # Discount factor
        'max_grad_norm': args.max_grad_norm,
        'rand_train_sizes': args.rand_train_sizes,

        # Test
        'max_test_steps': args.max_test_steps,
        'topk': args.topk,
        'n_solutions': args.n_solutions,
        'test_patience': args.test_patience,

        # Main directory
        'model_load': {
            'enable': args.model_load_enable,  # enable loading pre-trained model
            'path': args.model_load_path,  # directory path of pre-trained model and log files saved.
        },
        'save_model': args.save_model,
        'save_model_freq': args.save_model_freq,
        'verbose': args.verbose,
    }

    return env_params, memory_params, model_params, optimizer_params, trainer_params





