import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import cprint
import utils.evaluators
import wandb


def train_src(config, dataloaders, valloaders, testloaders, model, print_freq=0, datasets_name=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.pretrain_lr,
                                momentum=config.pretrain_momentum,
                                weight_decay=config.pretrain_weight_decay)

    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators_val = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in testloaders]


    for epoch in range(config.pretrain_epochs):
        model.train()
        for i, (imgs_src, labels_src) in enumerate(dataloaders[0]):
            imgs = imgs_src.to(config.device)
            labels = labels_src.to(config.device)
            labels = torch.argmax(labels, dim=1)
            output = model(imgs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        acc1s, _ = utils.evaluators.eval_func(config, evaluators_val, model)
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()

        wandb_log_dict = {
            'pt_epoch': epoch,
            'pt_loss': loss.item(),
            # 'pt_loss_val_ce': val_celoss[0].item(),
            'pt_Acc_val_src': acc1s[0]}

        # print validation
        print('[SL pretrain] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), acc1s[0]), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'pt_Acc_val_tgt_{dname}'] = acc_tgt
        # tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        wandb_log_dict[f'pt_Acc_val_tgt_mean'] = tgt_mean
        print(f'| tgt_mean: {tgt_mean:.2f}')

        wandb.log(wandb_log_dict)

    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)

    # wandb.run.summary['final_valbest_Acc_ft_src'] = bestlogger.result()['src']
    # wandb.run.summary['final_valbest_Acc_ft_tgtmean'] = bestlogger.result()['tgt']
    wandb.run.summary['pt_Acc_test_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'pt_Acc_test_tgt_{dname}'] = acc_tgt
    test_acc1s_tgt_mean = torch.mean(torch.tensor(test_acc1s[1:])).item()
    wandb.run.summary[f'pt_Acc_test_tgt_mean'] = test_acc1s_tgt_mean
    
    return test_acc1s[0], test_acc1s_tgt_mean
    


def eval_src(config, valloaders, model, print_freq=0, datasets_name=None):
    # YONGLI add: for colorMnist dataset
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]

    acc1s = []
    acc5s = []
    for evaluator in evaluators:
        eval_results = evaluator(model, device=config.device)
        if topk == (1, 5):
            (acc1, acc5), val_celoss_ = eval_results['Acc'], eval_results['Loss']
        elif topk == (1, ):
            acc, val_celoss_ = eval_results['Acc'], eval_results['Loss']
            acc1, acc5 = acc[0], 0.0
        acc1s.append(acc1)
        acc5s.append(acc5)

    print('[Evaluate] | src_acc1: %.1f, tgt_acc1: ' %
            (acc1s[0]), end='')
    for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
        print(f'{dname}: {acc_tgt:.2f} ', end='')
    tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
    print(f'| tgt_mean: {tgt_mean:.2f}')
    wandb_log_dict = {'pt_Acc_test_src': acc1s[0], 'pt_Acc_test_tgt_mean': tgt_mean}
    wandb.log(wandb_log_dict)
    
    return acc1s[0], tgt_mean

