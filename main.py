from functools import partial

from torch import save

#from fastai.basic_train import Learner
#from fastai.train import ShowGraph
#from fastai.data_block import DataBunch
from torch import optim

import torch #为了使用mps

import torch.nn as nn
from fastai.learner import Learner
from fastai.callback.progress import ShowGraphCallback
from fastai.data.core import DataLoaders

from fastai.optimizer import SGD

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss


# 检查MPS是否可用并设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#import os
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
#device = torch.device("cpu")

def main(args):
    # 实例化模型并将其移动到MPS设备
    model = UNet(1, 1, first_out_channels=16).to(device)   #使用Unet
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    batch_size = 4
    num_workers = 4
    optimizer = SGD
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    # windows上使用cuda
    #model = UNet(1, 1, first_out_channels=16)
    #model = nn.DataParallel(model.cuda())

    # 实例化模型并将其移动到MPS设备
    #model = UNet(1, 1, first_out_channels=16).to(device)

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)
    
    #databunch = DataBunch(dl_train, dl_val, collate_fn=FracNetTrainDataset.collate_fn)
    #databunch = DataLoaders(dl_train, dl_val, collate_fn=FracNetTrainDataset.collate_fn)
    # 更新DataLoaders的使用
    dls = DataLoaders(dl_train, dl_val)


    learn = Learner(
        dls,
        model,
        opt_func=partial(optimizer, lr=1e-1, mom=0.9),
        loss_func=criterion,
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial]
    )

    # 确保Learner使用MPS设备
    learn.model.to(device)
    
    #learn.fit_one_cycle(
    #用于训练模型
    learn.fit(
        n_epoch=200,    #训练过程将遍历整个数据集200次
        lr=1e-1,        #是学习率（Learning Rate）的设定，1e-1等于0.1。学习率是一个超参数，决定了模型学习的速度。如果学习率太大，模型可能会在最佳解周围震荡而无法收敛；如果学习率太小，训练过程可能会非常慢。
        #200,
        #1e-1,
        #pct_start=0,
            #pct_start=0.3,
        #div_factor=1000,
            #div=25.0,
        #callbacks=[ShowGraphCallback()]
        cbs=[ShowGraphCallback()]
    )

    if args.save_model:
        # cuda版本 多GYU并行处理-DataParallel版本的模型
        #save(model.module.state_dict(), "./model_weights.pth")
        # 因mps，故保存模型时确保保存的是非DataParallel版本的模型
        save(model.state_dict(), "./model_weights.pth")


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True,
        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", required=True,
        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", required=True,
        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", required=True,
        help="The validation label nii directory.")
    parser.add_argument("--save_model", default=False,
        help="Whether to save the trained model.")
    args = parser.parse_args()

    main(args)
