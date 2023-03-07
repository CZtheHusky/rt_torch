from rt_torch.utilizes.config import get_parser_for_basic_args, str2bool
import deepspeed


def parse_args():
    parser = get_parser_for_basic_args()
    parser = _add_training_args(parser)
    parser = _add_regularization_args(parser)
    # parser = _add_logging_args(parser)
    # parser = _add_checkpointing_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_deepspeed_args(parser)
    # parser = _add_finetune_args(parser)

    args = parser.parse_args()

    if args.weight_decay_incr_style == "constant":
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    if args.save_interval == None:
        args.save_interval = args.eval_interval

    return args


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Batch size per model instance (local batch size). "
        "Global batch size is local batch size times data "
        "parallel size times number of micro batches.",
    )
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Training batch size. If set, it should be a "
        "multiple of micro-batch-size times data-parallel-size. "
        "If this value is None, then "
        "use micro-batch-size * data-parallel-size as the "
        "global batch size. This choice will result in 1 for "
        "number of micro-batches.",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Total number of iterations to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )

    group.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Total number of iterations to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )

    group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer function",
    )


    group.add_argument(
        "--lr-decay-style",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine"],
        help="Learning rate decay function.",
    )

    group.add_argument(
        "--lr-decay-iters",
        type=int,
        default=None,
        help="number of iterations to decay learning rate over,"
        " If None defaults to `--train-iters`",
    )

    group.add_argument(
        "--lr-decay-samples",
        type=int,
        default=None,
        help="number of samples to decay learning rate over,"
        " If None defaults to `--train-samples`",
    )
    group.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=None,
        help="fraction of lr-warmup-(iters/samples) to use " "for warmup (as a float)",
    )
    group.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=0,
        help="number of iterations to linearly warmup " "learning rate over.",
    )
    group.add_argument(
        "--lr-warmup-samples",
        type=int,
        default=0,
        help="number of samples to linearly warmup " "learning rate over.",
    )
    group.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Old lr warmup argument, do not use. Use one of the"
        "--lr-warmup-* arguments above",
    )
    group.add_argument(
        "--override-opt_param-scheduler",
        action="store_true",
        help="Reset the values of the scheduler (learning rate,"
        "warmup iterations, minimum learning rate, maximum "
        "number of iterations, and decay style from input "
        "arguments and ignore values from checkpoints. Note"
        "that all the above values will be reset.",
    )
    group.add_argument(
        "--use-checkpoint-opt_param-scheduler",
        action="store_true",
        help="Use checkpoint to set the values of the scheduler "
        "(learning rate, warmup iterations, minimum learning "
        "rate, maximum number of iterations, and decay style "
        "from checkpoint and ignore input arguments.",
    )

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title="regularization")
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--start-weight-decay",
        type=float,
        help="Initial weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--end-weight-decay",
        type=float,
        help="End of run weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--weight-decay-incr-style",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Weight decay increment function.",
    )
    group.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping based on global L2 norm.",
    )
    group.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="First coefficient for computing running averages "
        "of gradient and its square",
    )
    group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="Second coefficient for computing running averages "
        "of gradient and its square",
    )
    group.add_argument(
        "--adam-eps",
        type=float,
        default=1e-08,
        help="Term added to the denominator to improve" "numerical stability",
    )
    group.add_argument(
        "--sgd-momentum", type=float, default=0.9, help="Momentum factor for sgd"
    )

    return parser




# def _add_logging_args(parser):
#     group = parser.add_argument_group(title="logging")
#     group.add_argument(
#         "--tensorboard-dir",
#         type=str,
#         default=None,
#         help="Write TensorBoard logs to this directory.",
#     )
#     group.add_argument(
#         "--tensorboard-queue-size",
#         type=int,
#         default=1000,
#         help="Size of the tensorboard queue for pending events "
#         "and summaries before one of the ‘add’ calls forces a "
#         "flush to disk.",
#     )
#     return parser


# def _add_checkpointing_args(parser):
#     group = parser.add_argument_group(title="checkpointing")
#     group.add_argument(
#         "--save-dir",
#         type=str,
#         default=None,
#         help="Output directory to save checkpoints to.",
#     )
#     group.add_argument(
#         "--save-interval",
#         type=int,
#         default=None,
#         help="Number of iterations between checkpoint saves.",
#     )
#     return parser


def _add_deepspeed_args(parser):
    group = parser.add_argument_group(title="_deepspeed_user_args")
    group.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher.",
    )
    group.add_argument(
        "--deepspeed_port", type=int, default=29500, help="Port to initialize deepspeed"
    )

    parser = deepspeed.add_config_arguments(parser)
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title="initialization")
    group.add_argument(
        "--init-method-std",
        type=float,
        default=0.02,
        help="Standard deviation of the zero mean normal "
        "distribution used for weight initialization.",
    )
    # parser.add_argument('--init-method-xavier-uniform', action='store_true',
    #                    help='Enable Xavier uniform parameter initialization')
    return parser

