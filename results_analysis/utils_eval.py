from collections import defaultdict

import copy
import numpy as np
import torch
import torch.nn.functional as F
import utils
# import sam


def attack_pgd(model, X, y, eps, alpha, scaler, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):

            with torch.cuda.amp.autocast():
                output = model(X + delta)  # + 0.25*torch.randn(X.shape).cuda())  # adding noise (aka smoothing)
                loss = F.cross_entropy(output, y)

            grad = torch.autograd.grad(scaler.scale(loss), delta)[0]
            grad = grad.detach() / scaler.get_scale()

            if not l2_grad_update:
                delta.data = delta + alpha * torch.sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = utils.clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = utils.clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=model.half_prec):
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = utils.clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta


def rob_err(batches, model, eps, pgd_alpha, scaler, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, verbose=False, cuda=True, noisy_examples='default', loss_f=F.cross_entropy,
            n_batches=-1):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    for i, (X, X_augm2, y, _, ln) in enumerate(batches):
        if n_batches != -1 and i > n_batches:  # limit to only n_batches
            break
        if cuda:
            X, y = X.cuda(), y.cuda()

        if noisy_examples == 'none':
            X, y = X[~ln], y[~ln]
        elif noisy_examples == 'all':
            X, y = X[ln], y[ln]
        else:
            assert noisy_examples == 'default'

        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, scaler, attack_iters, n_restarts, rs=rs,
                               verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=model.half_prec):
            output = model(X + pgd_delta)
            loss = loss_f(output, y)

        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)

    return 1 - robust_acc, avg_loss, pgd_delta_np



def norm_weights(model, ignore_bn=False):
    dist = 0
    for p_name, p1 in model.named_parameters():
        if ignore_bn and 'bn' in p_name:
            continue
        dist += torch.sum((p1.flatten())**2)
    dist = dist**0.5
    return dist.item()


def dist_models(model1, model2, ignore_bn=False):
    dist = 0
    for (p1_name, p1), (p2_name, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert p1_name == p2_name
        if ignore_bn and 'bn' in p1_name:
            continue
        dist += torch.sum((p1.flatten() - p2.flatten())**2)
    dist = dist**0.5
    return dist.item()


def step_size_schedule(init_step_size_linf, frac_iters):
    if frac_iters <= 0.25:
        return init_step_size_linf
    elif frac_iters <= 0.50:
        return init_step_size_linf / 2
    elif frac_iters <= 0.75:
        return init_step_size_linf / 4
    else:
        return init_step_size_linf / 8


# def eval_sharpness(model, batches, loss_f, rho, step_size, n_iters, n_restarts, apply_step_size_schedule=False, no_grad_norm=False, layer_name_pattern='all', random_targets=False, batch_transfer=False, rand_init=False, verbose=False):
#     orig_model_state_dict = copy.deepcopy(model.state_dict())

#     n_batches, best_obj_sum, final_err_sum, final_grad_norm_sum = 0, 0, 0, 0
#     for i_batch, (x, _, y, _, _) in enumerate(batches):
#         x, y = x.cuda(), y.cuda()

#         def f(model):
#             obj = loss_f(model(x), y)
#             return obj

#         obj_orig = f(model).detach()  
#         err_orig = (model(x).max(1)[1] != y).float().mean().item()

#         delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}
#         orig_param_dict = {param: param.clone() for param in model.parameters()}
#         best_obj, final_err, final_grad_norm = 0, 0, 0
#         for restart in range(n_restarts):
#             # random init on the sphere of radius `rho`
#             if rand_init:
#                 delta_dict = sam.random_init_on_sphere_delta_dict(delta_dict, rho)
#                 for param in model.parameters():
#                     param.data += delta_dict[param]
#             else:
#                 delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}

#             if rand_init:
#                 n_cls = 10
#                 y_target = torch.clone(y)
#                 for i in range(len(y_target)):
#                     lst_classes = list(range(n_cls))
#                     lst_classes.remove(y[i])
#                     y_target[i] = np.random.choice(lst_classes)
#             def f_opt(model):
#                 if not rand_init:
#                     return f(model)
#                 else:
#                     return -loss_f(model(x), y_target)

#             for iter in range(n_iters):
#                 step_size_curr = step_size_schedule(step_size, iter / n_iters) if apply_step_size_schedule else step_size
#                 delta_dict = sam.weight_ascent_step(model, f_opt, orig_param_dict, delta_dict, step_size_curr, rho, layer_name_pattern, no_grad_norm=no_grad_norm, verbose=False)
            
#             if batch_transfer:
#                 delta_dict_loaded = torch.load('deltas/gn_erm/batch{}.pth'.format(restart))  
#                 delta_dict_loaded = {param: delta for param, delta in zip(model.parameters(), delta_dict_loaded.values())}  # otherwise `param` doesn't work directly as a key
#                 for param in model.parameters():
#                     param.data = orig_param_dict[param] + delta_dict_loaded[param]

#             obj, grad_norm = utils.eval_f_val_grad(model, f)
#             err = (model(x).max(1)[1] != y).float().mean().item()

#             # if obj > best_obj:
#             if err > final_err:
#                 best_obj, final_err, final_grad_norm = obj, err, grad_norm
#             model.load_state_dict(orig_model_state_dict)
#             if verbose:
#                 delta_norm_total = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
#                 print('[restart={}] Sharpness: obj={:.4f}, err={:.2%}, delta_norm={:.2f} (step={:.3f}, rho={}, n_iters={})'.format(
#                       restart+1, obj - obj_orig, err - err_orig, delta_norm_total, step_size, rho, n_iters))

#         best_obj, final_err = best_obj - obj_orig, final_err - err_orig  # since we evaluate sharpness, i.e. the difference in the loss
#         best_obj_sum, final_err_sum, final_grad_norm_sum = best_obj_sum + best_obj, final_err_sum + final_err, final_grad_norm_sum + final_grad_norm
#         n_batches += 1

    
#     if type(best_obj_sum) == torch.Tensor:
#         best_obj_sum = best_obj_sum.item()
#     if type(final_grad_norm_sum) == torch.Tensor:
#         final_grad_norm_sum = final_grad_norm_sum.item()
        
#     return best_obj_sum / n_batches, final_err_sum / n_batches, final_grad_norm_sum / n_batches
