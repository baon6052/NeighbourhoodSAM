import argparse
import os
import copy
import numpy as np
import torch
import time
import sys
import wandb
from pathlib import Path

sys.path.append('../')



from args import parser
from datasets.dataset import get_dataset
from models.gcn import GCN
# import losses
import utils_eval
import utils_train

def main(args):
    start_time = time.time()

    print_stats = False
    n_cls = 2 if args.dataset in ['cifar10_horse_car', 'cifar10_dog_cat'] else 10
    p_label_noise = args.model_path_erm.split('p_label_noise=')[1].split(' ')[0]
    assert p_label_noise == args.model_path_sam.split('p_label_noise=')[1].split(' ')[0], 'ln level should be the same for the visualization'
    assert args.n_eval_sharpness % args.bs_sharpness == 0, 'args.n_eval should be divisible by args.bs_sharpness'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    datamodule = get_dataset(args.dataset_type, args.dataset_name, args.fold_idx, args.batch_size,
        args.neighbour_loader, args.num_neighbours)
    
    checkpoint_reference_ = args.checkpoint_reference_1
    checkpoint_reference = args.checkpoint_reference_2

    # download checkpoint locally (if not already cached)
    run1 = wandb.init(project=args.wandb_project, job_type="download_checkpoint")
    artifact1 = run1.use_artifact(args.checkpoint_reference_1, type="model")
    artifact_dir1 = artifact1.download()
    model_init1 = GCN.load_from_checkpoint(Path(artifact_dir1) / "model.ckpt")
    model_interpolated = GCN.load_from_checkpoint(Path(artifact_dir1) / "model.ckpt")
    
    run2 = wandb.init(project=args.wandb_project, job_type="download_checkpoint")
    artifact2 = run2.use_artifact(args.checkpoint_reference_2, type="model")
    artifact_dir2 = artifact2.download()
    model_init2 = GCN.load_from_checkpoint(Path(artifact_dir2) / "model.ckpt")

    # model_init1 = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
    # model_init1.apply(models.init_weights(args.model))
    # model_init2 = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
    # model_init2.apply(models.init_weights(args.model))
    # model_interpolated = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
    # model_interpolated.apply(models.init_weights(args.model))
    # model_erm = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
    # model_erm_swa = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
    # model_sam = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
    # model_sam_swa = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()

    # model_erm_dict_orig = torch.load('models/{}.pth'.format(args.model_path_erm))
    # model_erm_dict = model_erm_dict_orig['best'] if args.early_stopped_model_erm else model_erm_dict_orig['last']
    # model_erm.load_state_dict({k: v for k, v in model_erm_dict.items()})
    # if 'swa_best' in model_erm_dict_orig:
    #     model_erm_swa_dict = model_erm_dict_orig['swa_best']  # if args.early_stopped_model_erm else model_erm_dict_orig['swa_last']
    #     model_erm_swa.load_state_dict({k: v for k, v in model_erm_swa_dict.items()})
    # else:
    #     print('no swa_best checkpoint found in the ERM model_dict')

    # model_sam_dict_orig = torch.load('models/{}.pth'.format(args.model_path_sam))
    # model_sam_dict = model_sam_dict_orig['best'] if args.early_stopped_model_sam else model_sam_dict_orig['last']
    # model_sam.load_state_dict({k: v for k, v in model_sam_dict.items()})
    # if 'swa_best' in model_sam_dict_orig:
    #     model_sam_swa_dict = model_sam_dict_orig['swa_best']  # if args.early_stopped_model_sam else model_sam_dict_orig['swa_last']
    #     model_sam_swa.load_state_dict({k: v for k, v in model_sam_swa_dict.items()})
    # else:
    #     print('no swa_best checkpoint found in the SAM model_dict')

    # std_weight_perturb = 0.05
    # model_erm_plus_rand, model_sam_plus_rand = copy.deepcopy(model_erm), copy.deepcopy(model_sam)
    # utils_train.perturb_weights(model_erm_plus_rand, std_weight_perturb, 0, 'gauss')
    # utils_train.perturb_weights(model_sam_plus_rand, std_weight_perturb, 0, 'gauss')

    # model_zero = copy.deepcopy(model_erm)
    # utils_train.set_weights_to_zero(model_zero)

    # models_dict = {
    #     'erm': model_erm, 'erm_swa': model_erm_swa, 'sam': model_sam, 'sam_swa': model_sam_swa,
    #     'init': model_init1, 'erm_rand': model_erm_plus_rand, 'sam_rand': model_sam_plus_rand, 'zero': model_zero
    # }
    

    train_batches_for_bn = datamodule.train_dataloader()
    eval_train_batches = datamodule.train_dataloader()
    eval_test_batches = datamodule.test_dataloader()


    with torch.no_grad():
        alpha_step = 0.05  # 0.05 is sufficient
        alpha_range = np.concatenate([np.arange(-1.0, 0.0, alpha_step), np.arange(0.0, 2.0+alpha_step, alpha_step)])
        train_losses, test_losses = np.zeros_like(alpha_range), np.zeros_like(alpha_range)
        train_errors, test_errors = np.zeros_like(alpha_range), np.zeros_like(alpha_range)
        model1, model2 = model_init1, model_init2
        for i, alpha in enumerate(alpha_range):
            for (p, p1, p2) in zip(model_interpolated.parameters(), model1.parameters(), model2.parameters()):
                p.data = (1 - alpha) * p1.data + alpha * p2.data  # alpha=0: first model, alpha=1: second model
            utils_train.bn_update(train_batches_for_bn, model_interpolated)

            train_errors[i], train_losses[i], _ = utils_eval.rob_err(eval_train_batches, model_interpolated, 0, 0, scaler, 0, 0)
            test_errors[i], test_losses[i], _ = utils_eval.rob_err(eval_test_batches, model_interpolated, 0, 0, scaler, 0, 0)
            print('alpha={:.2f}: loss={:.3}/{:.3}, err={:.2%}/{:.2%}'.format(
                alpha, train_losses[i], test_losses[i], train_errors[i], test_errors[i]))


    export_dict = {'model_name1': args.model_name1, 'model_name2': args.model_name2,
                'alpha_range': alpha_range,
                'train_losses': train_losses, 'test_losses': test_losses,
                'train_errors': train_errors, 'test_errors': test_errors,
                'model1_norm': utils_eval.norm_weights(model_init1),
                'model2_norm': utils_eval.norm_weights(model_init2),
                # 'model1_sharpness_obj': model1_sharpness_obj, 'model2_sharpness_obj': model2_sharpness_obj,
                # 'model1_sharpness_grad_norm': model1_sharpness_grad_norm, 'model2_sharpness_grad_norm': model2_sharpness_grad_norm,
                }
    np.save('metrics_loss_surface/dataset={} ln={} early_stopped_model={}-{} models={}-{} n_eval={} label={}.npy'.format(
        args.dataset, p_label_noise, args.early_stopped_model_erm, args.early_stopped_model_sam, args.model_name1,
        args.model_name2, args.n_eval, args.label),
        export_dict)
    time_elapsed = time.time() - start_time
    print('Done in {:.2f}m'.format((time.time() - start_time) / 60))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
