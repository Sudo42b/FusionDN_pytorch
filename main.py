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
from models import MODEL, IQA_W, IQA_EN, IQA


def main(configs):
    if configs['task'] == 'vis_ir':
        return 0
    elif configs['task'] == 'oe_ue':
        return 1
    elif configs['task'] == 'far_near':
        return 2
    else:
        raise ValueError('task should be one of vis_ir, oe_ue, far_near')

from torch import nn
from torch.utils.data import DataLoader
def train_task(model:nn.Module, dataloader:DataLoader, 
               W_EN, c, configs, task):
    IQA_MODEL_PATH = configs['EXP_SETTING']['IQA_MODEL_PATH']
    n_batches = configs['MODEL']['BATCH_SIZE']
    W=configs['MODEL']['W_EN'],
    C=configs['MODEL']['C'],
    lam=configs['MODEL']['LAM']
    print('Train images number %d, Batches: %d.\n' % (len(dataloader), n_batches))
    if task == 'vis_ir':
        optimizer = RMSprop(model.parameters(), 
                            lr=configs['MODEL']['LEARNING_RATE'],
                            weight_decay=0.6, 
                            momentum=0.15)
    else:
        optimizer = RMSprop(model.parameters(), 
                            lr=configs['MODEL']['LEARNING_RATE'],
                            weight_decay=0.6, 
                            momentum=0.15)
        ewc_loss = 0
    # content_loss, ssim_loss, perceptual_Loss, gradient_Loss
    fusion_loss = Fusion_loss(batch_size=n_batches).to(configs['EXP_SETTING']['device'])
    
    ''' Perceptual loss'''
    model = model.to(configs['EXP_SETTING']['device'])
    theta = [p for p in model.parameters() if p.requires_grad]
    model.clip = [p.data.clamp_(-30, 30) for p in theta]
    
    #화면출력...
    loss_meter = AverageMeter(name= 'loss', fmt=':2.3e')
    #화면에 표시가 되야됨.
    progress_bar = ProgressMeter(num_batches=len(dataloader.dataset)//n_batches,
                                 meters=[loss_meter],
                                 prefix='Train: ')
    logger = Logger(path=os.path.join(configs['EXP_SETTING']['log_path'], 
                                f"{TASK}_{configs['MODEL']['LAM']}"),
                    header=['epoch','iteration','ssim', 'perceptual', 'gradient', 'content'],
                    resume=False)
    ACC, LOSS = {}, {}
    ACC[TASK], LOSS[TASK] = [], []
    for i in range(1, epoch+1, 1):
        start = time.time()
        for idx, data in enumerate(loader):
            
            #B, C(2), H, W
            source1_batch = data[:, 0, :, :].to(configs['EXP_SETTING']['device'])
            source2_batch = data[:, 1, :, :].to(configs['EXP_SETTING']['device'])
            
            optimizer.zero_grad()
            W = IQA_W(inputs1 = source1_batch, 
                        inputs2 = source2_batch, 
                        trained_model_path=IQA_MODEL_PATH,
                        w_en=W_EN,
                        c=c)
            
            S1_FEAS, S2_FEAS, F_FEAS, generator_img = model(I1 = source1_batch,
                                                            I2 = source2_batch)
            
           
            iqa_f = IQA(inputs=generator_img,
                        trained_model_path=IQA_MODEL_PATH)
            en_f = IQA_EN(inputs=generator_img,
                          patch_size=configs['MODEL']['PATCH_SIZE'])
            fusion_loss.W1 = W['W1'].to(configs['EXP_SETTING']['device'])
            fusion_loss.W2 = W['W2'].to(configs['EXP_SETTING']['device'])
            ssim_loss, per_loss, grad_loss, content_loss = fusion_loss(SOURCE1=source1_batch,
                                                            SOURCE2=source2_batch,
                                                            S1_FEAS=S1_FEAS,
                                                            S2_FEAS=S2_FEAS,
                                                            F_FEAS=F_FEAS,
                                                            generated_img=generator_img)
            
            loss = torch.mean(content_loss + iqa_f + W_EN * en_f)
            
            theta_G = [p for p in model.generator.parameters() if p.requires_grad]
            model.generator.clip_G = [p.data.clamp_(-8, 8) for p in theta_G]
            
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item())
            progress_bar.display(idx)
            logger.log({'epoch':i,'iteration':f'{idx}/{len(dataloader.dataset)//n_batches}' ,'ssim':ssim_loss.item(), 'perceptual':per_loss.item(),
                        'gradient':grad_loss.item(), 'content':content_loss.item()})
        print(f'epoch: {i} time: {time.time()-start}')
        
    return LOSS, ACC, model.state_dict()

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
    
    epoch = configs['MODEL'][TASK]['C']

    model = MODEL(batch_size=configs['MODEL']['BATCH_SIZE'],
                  input_size=configs['MODEL']['INPUT_SIZE'],
                  patch_size=configs['MODEL']['PATCH_SIZE'])
    
    loss, acc, weight = train_task(model=model, 
               dataloader=loader, 
               W_EN=configs['MODEL'][TASK]['W_EN'],
               c=configs['MODEL'][TASK]['C'],
               configs=configs,
               task=TASK)
    
    # This is for EWC, but IR-VR fusion model cannot not be used for EWC.
    # TODO : Implement EWC for Another dataset.
    assert TASK != 'vis_ir', 'IR-VR fusion model cannot be used for EWC. Further implementation is needed.'
    from train import ewc_process
    loss, acc = ewc_process(model=model, 
                epochs=epoch, 
                num_task=3,
                train_loader=loader,
                test_loader=loader,
                importance=1000, 
                weight=weight)