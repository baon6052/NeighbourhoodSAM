import torch
import numpy as np
import math

def perturb_weights(model, add_weight_perturb_scale, mul_weight_perturb_scale, weight_perturb_distr):
    with torch.no_grad():
        weights_delta_dict = {}
        for param in model.parameters():
            if param.requires_grad:
                if weight_perturb_distr == 'gauss':
                    delta_w_add = add_weight_perturb_scale * torch.randn(param.shape).cuda()  # N(0, std)
                    delta_w_mul = 1 + mul_weight_perturb_scale * torch.randn(param.shape).cuda()  # N(1, std)
                elif weight_perturb_distr == 'uniform':
                    delta_w_add = add_weight_perturb_scale * (torch.rand(param.shape).cuda() - 0.5)  # U(-0.5, 0.5)
                    delta_w_mul = 1 + mul_weight_perturb_scale * (torch.rand(param.shape).cuda() - 0.5)  # U(1 - 0.5*scale, 1 + 0.5*scale)
                else:
                    raise ValueError('wrong weight_perturb_distr')
                param_new = delta_w_mul * param.data + delta_w_add
                delta_w = param_new - param.data
                param.add_(delta_w)
                weights_delta_dict[param] = delta_w  # only the ref to the `param.data` is used as the key
    return weights_delta_dict


def set_weights_to_zero(model):
    for param in model.parameters():
        param.data = torch.zeros_like(param)


def subtract_weight_delta(model, weights_delta_dict):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.sub_(weights_delta_dict[param])  # get back to `w` from `w + delta`


def modify_grads_lin_model(model, x, y, eps, args):
    y_plus_minus = 2 * (y - 0.5)
    with torch.no_grad():  # completely override the gradient
        X_reshaped = x.reshape(x.shape[0], -1)
        w = model._model[1].weight
        w_l2_norm = (w ** 2).sum() ** 0.5
        if args.alpha_weights_linear_at:
            exp = torch.exp(-y_plus_minus[:, None] * X_reshaped @ w.T + eps * w_l2_norm)
            alphas = exp / (1 + exp)
            model._model[1].weight.grad = torch.mean(-alphas * y_plus_minus[:, None] * X_reshaped, 0, keepdim=True)
        if args.alpha_weights_linear_at1:
            exp = torch.exp(-y_plus_minus[:, None] * X_reshaped @ w.T)
            alphas = exp / (1 + exp)
            eps_stability = 0.00001
            model._model[1].weight.grad = torch.mean(
                -alphas * y_plus_minus[:, None] * X_reshaped + eps * w / (w_l2_norm + eps_stability), 0, keepdim=True)
        if args.alpha_weights_linear_at2:
            exp = torch.exp(-y_plus_minus[:, None] * X_reshaped @ w.T)
            alphas = exp / (1 + exp)
            eps_stability = 0.00001
            model._model[1].weight.grad = torch.mean(
                -alphas * y_plus_minus[:, None] * X_reshaped + alphas * eps * w / (w_l2_norm + eps_stability), 0,
                keepdim=True)


def change_bn_mode(model, bn_train):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if bn_train:
                module.train()
            else:
                module.eval()


def moving_average(net1, net2, alpha=0.999):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    with torch.no_grad():
        model.train()
        momenta = {}
        model.apply(reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for x, _, _, _, _ in loader:
            x = x.cuda(non_blocking=True)
            b = x.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(x)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.eval()

