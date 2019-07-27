# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, torch, datetime, shutil, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from openvqa.models.model_loader import ModelLoader
from openvqa.utils.optim import get_optim, adjust_lr
from utils.test_engine import test_engine, ckpt_proc


def train_engine(__C, dataset, dataset_eval=None):

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.train()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    # Define Loss Function
    loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")

    # Load checkpoint if resume training
    if __C.RESUME:
        print(' ========== Resume training')

        if __C.CKPT_PATH is not None:
            print('Warning: Now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = __C.CKPT_PATH
        else:
            path = __C.CKPTS_PATH + \
                   '/ckpt_' + __C.CKPT_VERSION + \
                   '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

        # Load the network parameters
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

        if __C.N_GPU > 1:
            net.load_state_dict(ckpt_proc(ckpt['state_dict']))
        else:
            net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

        # Load the optimizer paramters
        optim = get_optim(__C, net, data_size, ckpt['lr_base'])
        optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(ckpt['optimizer'])
        
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

    else:
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            #shutil.rmtree(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

        optim = get_optim(__C, net, data_size)
        start_epoch = 0

    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))

    # Define multi-thread dataloader
    # if __C.SHUFFLE_MODE in ['external']:
    #     dataloader = Data.DataLoader(
    #         dataset,
    #         batch_size=__C.BATCH_SIZE,
    #         shuffle=False,
    #         num_workers=__C.NUM_WORKERS,
    #         pin_memory=__C.PIN_MEM,
    #         drop_last=True
    #     )
    # else:
    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM,
        drop_last=True
    )

    logfile = open(
        __C.LOG_PATH +
        '/log_run_' + __C.VERSION + '.txt',
        'a+'
    )
    logfile.write(str(__C))
    logfile.close()

    # Training script
    for epoch in range(start_epoch, __C.MAX_EPOCH):

        # Save log to file
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            '=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()

        # Learning Rate Decay
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)

        # Externally shuffle data list
        # if __C.SHUFFLE_MODE == 'external':
        #     dataset.shuffle_list(dataset.ans_list)

        time_start = time.time()
        # Iteration
        for step, (
                frcn_feat_iter,
                grid_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):

            optim.zero_grad()

            frcn_feat_iter = frcn_feat_iter.cuda()
            grid_feat_iter = grid_feat_iter.cuda()
            bbox_feat_iter = bbox_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            ans_iter = ans_iter.cuda()

            loss_tmp = 0
            for accu_step in range(__C.GRAD_ACCU_STEPS):
                loss_tmp = 0

                sub_frcn_feat_iter = \
                    frcn_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_grid_feat_iter = \
                    grid_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_bbox_feat_iter = \
                    bbox_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ques_ix_iter = \
                    ques_ix_iter[accu_step * __C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ans_iter = \
                    ans_iter[accu_step * __C.SUB_BATCH_SIZE:
                             (accu_step + 1) * __C.SUB_BATCH_SIZE]

                pred = net(
                    sub_frcn_feat_iter,
                    sub_grid_feat_iter,
                    sub_bbox_feat_iter,
                    sub_ques_ix_iter
                )

                loss_item = [pred, sub_ans_iter]
                loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat']:
                        loss_item[item_ix] = loss_item[item_ix].view(-1)
                    elif loss_nonlinear:
                        loss_item[item_ix] = eval('F.' + loss_nonlinear + '(loss_item[item_ix], dim=1)')

                loss = loss_fn(loss_item[0], loss_item[1])
                loss /= __C.GRAD_ACCU_STEPS
                loss.backward()

                loss_tmp += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS

            if __C.VERBOSE:
                if dataset_eval is not None:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
                else:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

                print("\r[Version %s][Model %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e" % (
                    __C.VERSION,
                    __C.MODEL_USE,
                    __C.DATASET,
                    epoch + 1,
                    step,
                    int(data_size / __C.BATCH_SIZE),
                    mode_str,
                    loss_tmp / __C.SUB_BATCH_SIZE,
                    optim._rate
                ), end='          ')

            # Gradient norm clipping
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )

            # Save the gradient information
            for name in range(len(named_params)):
                norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                    if named_params[name][1].grad is not None else 0
                grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS
                # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                #       (str(grad_wt),
                #        params[grad_wt][0],
                #        str(norm_v)))

            optim.step()

        time_end = time.time()
        elapse_time = time_end-time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Save checkpoint
        if __C.N_GPU > 1:
            state = {
                'state_dict': net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        else:
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        torch.save(
            state,
            __C.CKPTS_PATH +
            '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) +
            '.pkl'
        )

        # Logging
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / data_size) +
            ', Lr: ' + str(optim._rate) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) + 
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )
        logfile.close()

        # Eval after every epoch
        if dataset_eval is not None:
            test_engine(
                __C,
                dataset_eval,
                state_dict=net.state_dict(),
                validation=True
            )

        # if self.__C.VERBOSE:
        #     logfile = open(
        #         self.__C.LOG_PATH +
        #         '/log_run_' + self.__C.VERSION + '.txt',
        #         'a+'
        #     )
        #     for name in range(len(named_params)):
        #         logfile.write(
        #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
        #                 str(name),
        #                 named_params[name][0],
        #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
        #             )
        #         )
        #     logfile.write('\n')
        #     logfile.close()

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
