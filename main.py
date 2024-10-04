"""

"""
import os
import torch
from torch.optim.rmsprop import RMSprop
import argparse
from configs.io import TOMLConfig
from datasets import get_dataloader
from utils.util import AverageMeter, ProgressMeter, reproducibility, accuracy
from torchvision import transforms as T
from utils.logger import Logger
import time
from losses import Fusion_loss
from models import MODEL
def train(config):
    pass

def test(config):
    pass

def main(configs):
    if configs['task'] == 'vis_ir':
        return 0
    elif configs['task'] == 'oe_ue':
        return 1
    elif configs['task'] == 'far_near':
        return 2
    else:
        raise ValueError('task should be one of vis_ir, oe_ue, far_near')



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        model = model.cuda()
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)

def evaluate(dataloader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(dataloader):
            model = model.cuda()
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

    return float(top1.avg), float(top5.avg)


def train_task(model, dataloader, configs):
    IQA_MODEL_PATH = configs['EXP_SETTING']['IQA_MODEL_PATH']
    n_batches = configs['MODEL']['BATCH_SIZE']
    W=configs['MODEL']['W_EN'],
    C=configs['MODEL']['C'],
    lam=configs['MODEL']['LAM']
    print('Train images number %d, Batches: %d.\n' % (len(dataloader), n_batches))
    criterion = RMSprop(model.parameters(), 
                        lr=configs['MODEL']['LEARNING_RATE'],
                        weight_decay=0.6, 
                        momentum=0.15)
    
    logger = Logger(os.path.join(configs['EXP_SETTING']['log_path'], 
                                 f"{TASK}_{configs['MODEL']['LAM']}"),
                    ['epoch', 'loss'],)
    
    loss = Fusion_loss(batch_size=n_batches)
    ''' Perceptual loss'''
    model = model.to(configs['EXP_SETTING']['device'])
    for i in range(1, epoch+1, 1):
        print(f'epoch: {i}')
        start = time.time()
        for idx, data in enumerate(loader):
            progress_bar.display(idx)
            source1_batch = data[:, 0, :, :].to(configs['EXP_SETTING']['device'])
            source2_batch = data[:, 1, :, :].to(configs['EXP_SETTING']['device'])
            S1_FEAS, S2_FEAS, F_FEAS, generator_img = model(I1 = source1_batch,
                                                            I2 = source2_batch)
            for s1, s2, f in zip(S1_FEAS, S2_FEAS, F_FEAS):
                print(s1.shape, s2.shape, f.shape)
            
            exit()
            

        print(f'epoch: {i} time: {time.time()-start}')

if __name__ =='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='configs/config.toml')
    args.add_argument('--task', type=str, default='vis_ir',
                      help='select one of tasks(vis_ir, oe_ue, far_near)')
    args = args.parse_args()
    configs = TOMLConfig(args.config)
    print(configs)
    TASK = args.task
    # main(configs)
    # source_data1 = h5py.File('./datasets/vis_ir_dataset64.h5', 'r')['data'] # (N, C, H, W)

    loader =get_dataloader(path = configs['EXP_SETTING'][TASK], 
                   batch_size=configs['MODEL']['BATCH_SIZE'],
                   device=configs['EXP_SETTING']['device'])
    
    reproducibility(configs['EXP_SETTING']['seed'])
    #로그쓰는거...
    content_loss = AverageMeter(name= 'content_loss', fmt=':.4e')
    #화면에 표시가 되야됨.
    progress_bar = ProgressMeter(num_batches=configs['MODEL']['BATCH_SIZE'],
                                 meters=[content_loss],
                                 prefix='Train: ')
    epoch = configs['MODEL'][TASK]['C']

    model = MODEL(batch_size=configs['MODEL']['BATCH_SIZE'],
                  input_size=configs['MODEL']['INPUT_SIZE'],
                  patch_size=configs['MODEL']['PATCH_SIZE'])
    
    train_task(model=model, 
               dataloader=loader, 
               configs=configs)
    
    import os
    
    # print(source_data1.shape)
    # tf.summary.scalar('content_Loss', model.content_loss)
    # tf.summary.scalar('ssim_Loss', model.ssim_loss)
    # tf.summary.scalar('perceptual_Loss', model.perloss)
    # tf.summary.scalar('gradient_Loss', model.grad_loss)
    # tf.summary.image('source1', model.SOURCE1, max_outputs = 3)
    # tf.summary.image('source2', model.SOURCE2, max_outputs = 3)
    # tf.summary.image('fused_result', model.generated_img, max_outputs = 3)
    
    
    
    # Save model save_path in args
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    # torch.save(model.state_dict(), os.path.join(os.getcwd(), args.save_path, "vww_96_vgg.pth"))
    
