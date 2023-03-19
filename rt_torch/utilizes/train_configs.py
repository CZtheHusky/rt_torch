from torch import nn
from torch.optim import Adam, SGD, AdamW
from rt_torch.utilizes.optimizer_param_scheduler import OptimizerParamScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_optimizer_param_scheduler(args, optimizer):
    """Build the learning rate scheduler."""

    # # Iteration-based training.
    # if args.lr_decay_iters is None:
    #     args.lr_decay_iters = args.train_iters
    # lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    # wd_incr_steps = args.train_iters * args.global_batch_size
    # # if args.lr_warmup_fraction is not None:
    # #     lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    # # else:
    # #     lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    # opt_param_scheduler = OptimizerParamScheduler(
    #     optimizer,
    #     max_lr=args.lr,
    #     min_lr=args.min_lr,
    #     lr_warmup_steps=0,
    #     lr_decay_steps=lr_decay_steps,
    #     lr_decay_style=args.lr_decay_style,
    #     start_wd=args.start_weight_decay,
    #     end_wd=args.end_weight_decay,
    #     wd_incr_steps=wd_incr_steps,
    #     wd_incr_style=args.weight_decay_incr_style,
    #     use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
    #     override_opt_param_scheduler=args.override_opt_param_scheduler,
    # )
    if args.lr_decay_style == "cosine":
        opt_param_scheduler = CosineAnnealingLR(optimizer, T_max=args.train_iters, eta_min=args.min_lr)
    else:
        raise NotImplementedError
    return opt_param_scheduler


def get_paramaters(args, model: nn.Module):
    lr_t = args.lr_t
    lr_eff = args.lr_eff
    assert lr_t == 1 or lr_eff == 1
    if lr_t == lr_eff:
        return model.parameters()
    pretrained = set()
    unpretrained = set()
    for module_name, module in model.named_modules():
        for parameter_name, paramater in module.named_parameters():
            full_parameter_name = "%s.%s" % (module_name, parameter_name) if module_name else parameter_name  # full param name
            if full_parameter_name.endswith("bias") or full_parameter_name.endswith("weight"):
                if "film_efficient_net" in full_parameter_name:
                    if "conditioning_layers" in full_parameter_name:
                        unpretrained.add(full_parameter_name)
                    else:
                        pretrained.add(full_parameter_name)
                else:
                    unpretrained.add(full_parameter_name)
            else:
                import pdb; pdb.set_trace()
    param_dict = {parameter_name: paramater for parameter_name, paramater in model.named_parameters()}
    pretrained = param_dict.keys() & pretrained
    unpretrained = param_dict.keys() & unpretrained
    inter_params = pretrained & unpretrained
    union_params = pretrained | unpretrained
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both pretrained/unpretrained sets!" % (str(inter_params),)
    assert len(param_dict.keys() ^ union_params) == 0, (
        "parameters %s were not separated into either pretrained/unpretrained set!"
        % (str(param_dict.keys() - union_params),)
    )
    optim_groups = [
        {
            "params": [
                param_dict[pn] for pn in sorted(list(pretrained)) if pn in param_dict
            ],
            "lr": lr_eff * args.lr,
        },
        {"params": [param_dict[pn] for pn in sorted(list(unpretrained))], "lr": lr_t * args.lr},
    ]
    return optim_groups
        

def get_optimizer(args, model):
    # Base optimizer.
    paramaters = get_paramaters(args, model)
    if args.optimizer == "adam":
        optimizer = Adam(
            paramaters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            paramaters,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            paramaters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
        )
    else:
        raise Exception("{} optimizer is not supported.".format(args.optimizer))

    return optimizer