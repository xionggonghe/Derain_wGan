import os
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import shutil
import random
import torch.optim as optim

from model.networks import Generator, Discriminator
from utils.data_RGB import get_training_data, get_validation_data
# from losses import SupConLoss
from utils.PSNR import torchPSNR as PSNR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

NUM_EPOCHS = 500  # 训练周期
patch_size = 128
BATCH_SIZE = 16
lr_G = 0.0001
lr_D = 0.0002
Weight = [0.05, 0.1, 0.2, 0.2, 0.2, 0.15, 0.15]
model_dir = "./result/"

# Loss function
adversarial_loss = []
adversarial_loss.append(nn.CosineSimilarity(dim=1).to(device))  # Vector Constrastive learning Loss
adversarial_loss.append(
    nn.BCEWithLogitsLoss(weight=None, reduction='mean').to(device))  # projecthead classifer Constrastive learning Loss
adversarial_loss.append(nn.L1Loss().to(device))  # feature Constrastive learning Loss
# adversarial_loss.append(SupConLoss(temperature=0.1))  # sup-Constrastive learning Loss
adversarial_loss.append(nn.L1Loss().to(device))

# Initialize generator and discriminator
generator = Generator().to(device)
cnn_paras_count(generator)
discriminator = Discriminator().to(device)
cnn_paras_count(discriminator)

# xADD = self.MixUp0[0](xADD, res[self.num_blocks])print(generator)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D)

######### Scheduler ###########
warmup_epochs = 3
schedulerGen = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=0.9, patience=15, verbose=True,
                threshold=0.01, threshold_mode='abs', cooldown=3, min_lr=0, eps=1e-08)
schedulerDis = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', factor=0.9, patience=15, verbose=True,
                threshold=0.01, threshold_mode='abs', cooldown=3, min_lr=0, eps=1e-08)
# schedulerGen = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
# schedulerDis = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)

# Data loading code
train_dir = "./dataset/rain_data_train_Light/"  # 训练数据集目录
train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                           drop_last=False, pin_memory=True)

# Data loading code
Vaild_dir = "./dataset/rain_data_test_Light/"  # 训练数据集目录
Vaild_dataset = get_validation_data(Vaild_dir, {'patch_size': patch_size})
Vaild_loader = torch.utils.data.DataLoader(dataset=Vaild_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                           drop_last=False, pin_memory=True)


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def Count_GenAdversarialLoss(discriminator, gen_imgs, Rain_img, Norain_img):
    Vector_gen, Classfier_gen, fea_gen = discriminator(gen_imgs)
    Vector_Rain, Classfier_Rain, fea_Rain = discriminator(Rain_img)
    Vector_Norain, Classfier_Norain, fea_Norain = discriminator(Norain_img)
    Loss_VectorP = adversarial_loss[0](Vector_gen, Vector_Rain).sum(0)  # CosineSimilarity
    Loss_VectorN = adversarial_loss[0](Vector_gen, Vector_Rain).sum(0)  # CosineSimilarity
    Loss_Vector = Loss_VectorP / (Loss_VectorP + Loss_VectorN)

    labels_True = torch.ones([Classfier_gen.shape[0]]).to(device)
    labels_False = torch.zeros([Classfier_gen.shape[0]]).to(device)
    labels_N = torch.stack([labels_True, labels_False], dim=1)  # 位置编码 前:无雨 后:有雨
    Loss_Classfier = adversarial_loss[1](Classfier_gen, labels_N)  # Softmax 0:无雨 1:有雨

    # Loss_Classfier_simsiam = -adversarial_loss[2]() + adversarial_loss[2]()
    # -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    Loss_feas = 0.0
    for i in range(len(fea_gen)):
        Loss_feas = Loss_feas + Weight[i] * adversarial_loss[2](fea_gen[i], fea_Norain[i])  # L1Loss

    Loss_L1 = adversarial_loss[3](gen_imgs, Norain_img)
    Loss = 0.5 * Loss_Vector + Loss_Classfier + 0.5 * Loss_feas + Loss_L1
    return Loss


def Count_DisAdversarialLoss(discriminator, gen_imgs, Rain_img, Norain_img):
    Vector_gen, Classfier_gen, fea_gen = discriminator(gen_imgs)
    Vector_Rain, Classfier_Rain, fea_Rain = discriminator(Rain_img)
    Vector_Norain, Classfier_Norain, fea_Norain = discriminator(Norain_img)
    Loss_VectorP = adversarial_loss[0](Vector_gen, Vector_Norain).sum(0)  # CosineSimilarity
    Loss_VectorN = adversarial_loss[0](Vector_gen, Vector_Rain).sum(0)  # CosineSimilarity
    Loss_Vector = Loss_VectorP / (Loss_VectorP + Loss_VectorN)

    labels_True = torch.ones([Classfier_gen.shape[0]]).to(device)
    labels_False = torch.zeros([Classfier_gen.shape[0]]).to(device)
    labels_R = torch.stack([labels_False, labels_True], dim=1)  # 位置编码 前:无雨 后:有雨
    labels_N = torch.stack([labels_True, labels_False], dim=1)  # 位置编码 前:无雨 后:有雨
    Loss_Classfier = adversarial_loss[1](Classfier_gen, labels_R) + adversarial_loss[1](Classfier_Rain, labels_R) + \
                     adversarial_loss[1](Classfier_Norain, labels_N)
    Loss_feas = 0.0
    for i in range(len(fea_gen)):
        Loss_feas = Loss_feas + Weight[i] * adversarial_loss[2](fea_gen[i], fea_Norain[i].detach())  # L1Loss

    # fea_SupDic = {0: Vector_Rain, 1: Vector_Norain, 2: Vector_gen}
    # index = [0, 1, 2]
    # random.shuffle(index)
    # fea_Sup = torch.cat([fea_SupDic[index[0]], fea_SupDic[index[1]], fea_SupDic[index[2]]], dim=0)
    # fea_Sup = torch.unsqueeze(fea_Sup, dim=1)
    # fea_labelsList = []
    # for i in range(len(index)):
    #     if index[i]%2 == 0:
    #         fea_labelsList.append(torch.zeros([Vector_Rain.shape[0]]))
    #     else:
    #         fea_labelsList.append(torch.ones([Vector_Rain.shape[0]]))
    # fea_labels = torch.cat([fea_labelsList[0], fea_labelsList[1], fea_labelsList[2]], dim=0).to(device)
    #
    # Loss_Sup = adversarial_loss[3](fea_Sup, fea_labels)

    Loss = Loss_Vector + Loss_Classfier + 0.5 * Loss_feas
    return Loss


for epoch in range(1, NUM_EPOCHS):
    # batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    # lossesGen = AverageMeter('lossesGen', ':.4f')
    # lossesDis = AverageMeter('lossesDis', ':.4f')
    Psnr = AverageMeter('psnr', ':.3f')
    Lr = AverageMeter('lr', ':.6f')

    progress = ProgressMeter(
        len(train_loader),
        # [batch_time, data_time, lossesGen, lossesDis, Psnr],
        [data_time, Psnr, Lr],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    psnr_tra_rgb = []
    for i, images in enumerate(tqdm(train_loader), 0):
        generator.train()
        discriminator.train()
        # measure data loading time
        data_time.update(time.time() - end)

        Norain_img = images[0].to(device)
        Rain_img = images[1].to(device)

        # -------------------------------------------------------------------------------------
        #  Train Generator
        # -------------------------------------------------------------------------------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(Rain_img)

        # Loss measures generator's ability to fool the discriminator
        g_loss = Count_GenAdversarialLoss(discriminator, gen_imgs, Rain_img, Norain_img)
        g_loss.backward(retain_graph=True)

        # lossesGen.update(g_loss.item(), images[0].size(0))
        Psnr.update(PSNR(Norain_img, gen_imgs.detach()).mean().item())

        # -----------------------------------------------------------------------------------------
        #  Train Discriminator
        # -----------------------------------------------------------------------------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        gen_img = gen_imgs.clone()
        d_loss = Count_DisAdversarialLoss(discriminator, gen_img, Rain_img, Norain_img)

        d_loss.backward()
        optimizer_G.step()
        optimizer_D.step()
        # lossesDis.update(d_loss.item(), images[0].size(0))
        # measure elapsed time
        # batch_time.update(time.time() - end)
        Lr.update(optimizer_G.state_dict()['param_groups'][0]['lr'])
        progress.display(i)

    # schedulerGen.step()
    # schedulerDis.step()


    with torch.no_grad():
        #### Evaluation ####
        if epoch % 1 == 0:
            generator.eval()
            discriminator.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((Vaild_loader), 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)

                with torch.no_grad():
                    restored = generator(input_)

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(PSNR(res, tar))
                psnr = PSNR(res, tar)


            if epoch % 10 == 0:
                torch.save({'state_dictG': generator.state_dict(),
                            'state_dictD': discriminator.state_dict(),
                            }, os.path.join(model_dir, "model2_{}.pth".format(epoch)))
    schedulerGen.step(psnr)
    schedulerDis.step(psnr)



    #
    # print("------------------------------------------------------------------")
    #     print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
    #                                                                               epoch_loss, scheduler.get_lr()[0]))
    #     print("------------------------------------------------------------------")
    #
    #     torch.save({'epoch': epoch,
    #                 'state_dict': model.state_dict(),
    #                 'optimizer': optimizer.state_dict()
    #                 }, os.path.join(model_dir, "model_latest.pth"))







