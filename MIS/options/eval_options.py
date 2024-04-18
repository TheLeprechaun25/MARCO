import argparse


def get_options():
    parser = argparse.ArgumentParser(description="Marco")

    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    # Environment
    parser.add_argument('--n', type=int, default=100, help='Problem size')
    parser.add_argument('--graph_type', type=str, default='er100', choices=['er{N}', 'er700800', 'rb200300', 'rb8001200'], help='Graph type to evaluate.')
    parser.add_argument('--initialization', type=str, default='greedy2', choices=['zeros', 'greedy', 'greedy2'], help='Initialization')

    # Memory
    parser.add_argument('--memory_type', type=str, default='shared', choices=['shared', 'individual', 'op_based', 'none'], help='Memory type: shared, individual, operation-based, none')
    parser.add_argument('--memory_size', type=int, default=10000, help='Memory size')
    parser.add_argument('--k', type=int, default=20, help='k')
    parser.add_argument('--mem_aggr', type=str, default='linear', choices=['sum', 'linear', 'exp'], help='The aggregation method for the memory. Sum of all k values. Linear weighted sum. Exponential weighted sum.')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--n_layers', type=int, default=3, help='n_layers')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
    parser.add_argument('--normalization', type=str, default='layer', choices=['batch', 'instance', 'layer', 'rms'], help='Normalization')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'swiglu', 'relu'], help='Activation')
    parser.add_argument('--bias', type=bool, default=False, help='Bias')
    parser.add_argument('--tanh_clipping', type=float, default=10, help='Tanh clipping')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')

    # Eval
    parser.add_argument('--n_instances', type=int, default=10, help='Number of instances to solve')
    parser.add_argument('--save_all_time', action='store_true', help='Save all time')
    parser.add_argument('--n_solutions', type=int, default=50, help='Number of threads')
    parser.add_argument('--topk', type=int, default=10, help='Executed actions in each inference step. topk=1 for default inference')
    parser.add_argument('--max_steps', type=int, default=2, help='Max test steps. 2 refers to 2*N')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    # Others
    parser.add_argument('--model_load_path', type=str, default='used_models/marco_checkpoint.pth', help='Model load path')
    parser.add_argument('--compile', action='store_true', help='Compile pytorch module')
    parser.add_argument('--save_results', type=bool, default=False, help='Save results')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose')

    args = parser.parse_args()
    # group all arguments in different dictionaries
    env_params = {
        'problem_size': args.n,
        'graph_type': args.graph_type,
        'initialization': args.initialization,
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
        'n_heads': args.n_heads,
        'normalization': args.normalization,
        'activation': args.activation,
        'bias': args.bias,
        'tanh_clipping': args.tanh_clipping,
        'dropout': args.dropout,
    }
    eval_params = {
        'n_instances': args.n_instances,
        'save_all_time': args.save_all_time,  # save all time or only the last time step
        'seed': args.seed,
        'n_solutions': args.n_solutions,
        'topk': args.topk,
        'max_steps': args.max_steps,
        'patience': args.patience,

        'model_load_path': args.model_load_path,  # directory path of pre-trained model and log files saved.
        'compile': args.compile,
        'save_results': args.save_results,
        'verbose': args.verbose,
    }

    return env_params, memory_params, model_params, eval_params
