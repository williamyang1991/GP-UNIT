import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
import math
from tqdm import tqdm
from model.content_encoder import AutoEncoder
from model.vgg import VGGLoss
from dataset import create_imagemasklabel_dataloader, natural_sort

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train Prior Distillation of GP-UNIT")
        self.parser.add_argument("--task", type=str, default='prior_distillation', help="task name")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
        self.parser.add_argument("--iter", type=int, default=45000, help="iterations")
        self.parser.add_argument("--batch", type=int, default=16, help="batch size")
        self.parser.add_argument("--lambda_app_rec", type=float, default=1.0, help="the weight of appearance distant loss")
        self.parser.add_argument("--lambda_shape_rec", type=float, default=5.0, help="the weight of shape reconstruction loss")
        self.parser.add_argument("--lambda_reg", type=float, default=0.001, help="the weight of regularization loss")
        self.parser.add_argument("--lambda_class", type=float, default=1.0, help="the weight of classification loss")
        self.parser.add_argument("--lambda_feat_dist", type=float, default=1.0, help="the weight of feature distant loss")
        self.parser.add_argument("--lambda_shape_dist", type=float, default=5.0, help="the weight of shape distant loss")
        self.parser.add_argument("--paired_data_root", type=str, help="the path to the synImageNet291")
        self.parser.add_argument("--unpaired_data_root", type=str, help="the path to the ImageNet291 and CelebA-HQ")
        self.parser.add_argument("--paired_mask_root", type=str, help="the path to the synImageNet291_mask")
        self.parser.add_argument("--unpaired_mask_root", type=str, help="the path to the ImageNet291_mask and CelebA-HQ_mask")
        self.parser.add_argument("--save_every", type=int, default=5000, help="interval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=30000, help="when to start saving a checkpoint")
        self.parser.add_argument("--visualize_every", type=int, default=500, help="interval of saving an intermediate result")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path to the saved models")  
        
    def parse(self):
        self.opt = self.parser.parse_args()       
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
    
def train(args, udataloader, pdataloader, netAE, optimizer_AE, vgg_loss, device='cuda'):
    pbar = tqdm(range(args.iter), initial=0, smoothing=0.01, ncols=120, dynamic_ncols=False)

    netAE.train()
    piterator = iter(pdataloader)
    uiterator = iter(udataloader)
    for idx in pbar:
        try:
            pdata = next(piterator)
        except StopIteration:
            piterator = iter(pdataloader)
            pdata = next(piterator) 
        try:
            udata = next(uiterator)
        except StopIteration:
            uiterator = iter(udataloader)
            udata = next(uiterator) 

        la, xa, lb, xb, ma, mb = pdata['label'], pdata['image'], pdata['labelB'], pdata['imageB'], pdata['mask'], pdata['maskB']
        l, x, m = udata['label'], udata['image'], udata['mask']
        la, lb, l = la.to(device), lb.to(device), l.to(device)
        xa, xb, x = xa.to(device), xb.to(device), x.to(device)
        m = F.interpolate(m.to(device), size=16, mode='bilinear')
        ma = F.interpolate(ma.to(device), size=16, mode='bilinear')
        mb = F.interpolate(mb.to(device), size=16, mode='bilinear')

        imgs = torch.cat((x,xa,xb), dim=0)
        labels = torch.cat((l,la,lb), dim=0)
        masks = torch.cat((m,ma,mb), dim=0)

        loss_dict = {}

        # encode with random flip
        noisy_content_feat, content_feat, style_feat = netAE.encode(imgs, 0.5)
        label_feat = netAE.embed(labels)
        recon_imgs, recon_masks = netAE.decode(style_feat, label_feat, noisy_content_feat)

        # exchange content features of xa and xb for translation 
        style_feat2 = []
        for s in style_feat:
            style_feat2.append([s[0][args.batch:args.batch*3],s[1][args.batch:args.batch*3]])
        _, trans_masks = netAE.decode(style_feat2, label_feat[args.batch:args.batch*3], 
                                 torch.cat((noisy_content_feat[args.batch*2:args.batch*3],
                                            noisy_content_feat[args.batch:args.batch*2]), dim=0))

        pred = netAE.classify(content_feat)

        Larec = (F.mse_loss(recon_imgs, imgs) + vgg_loss(recon_imgs, imgs)) * args.lambda_app_rec
        Lsrec = F.l1_loss(recon_masks, masks) * args.lambda_shape_rec
        Lfdist = F.l1_loss(content_feat[args.batch:args.batch*2], content_feat[args.batch*2:args.batch*3].detach()) * args.lambda_feat_dist
        Lsdist = F.l1_loss(trans_masks, masks[args.batch:args.batch*3]) * args.lambda_shape_dist
        Lreg = torch.norm(content_feat, p=2) * args.lambda_reg + F.cross_entropy(pred, labels) * args.lambda_class

        loss_dict['arec'] = Larec
        loss_dict['srec'] = Lsrec
        loss_dict['fdist'] = Lfdist
        loss_dict['sdist'] = Lsdist
        loss_dict['reg'] = Lreg

        ae_loss = Larec + Lsrec + Lfdist + Lsdist + Lreg

        optimizer_AE.zero_grad()  
        ae_loss.backward()
        optimizer_AE.step()        

        message = ''
        for k, v in loss_dict.items():
            v = v.mean().float()
            message += 'L%s: %.3f ' % (k, v)
        pbar.set_description((message))

        if ((idx+1) >= args.save_begin and (idx+1) % args.save_every == 0)  or (idx+1) == args.iter:
            torch.save(
                {
                    "ae_ema": netAE.state_dict(),
                    "ae_optim": optimizer_AE.state_dict(),
                    #"args": args,
                },
                f"%s/%s-%05d.pt"%(args.model_path, args.task, idx+1),
            )
            if (idx+1) == args.iter:
                torch.save(netAE.contentE.state_dict(),f"%s/content_encoder-%05d.pt"%(args.model_path, idx+1))


        if idx == 0 or (idx+1) % args.visualize_every == 0 or (idx+1) == args.iter:
            viznum = min(args.batch, 4)
            masks = F.interpolate(masks, size=256)
            recon_masks = F.interpolate(recon_masks, size=256)
            trans_masks = F.interpolate(trans_masks, size=256)

            sample = F.adaptive_avg_pool2d(torch.cat((imgs[0:viznum], masks[0:viznum],
                                                    recon_imgs[0:viznum], recon_masks[0:viznum],
                                                    imgs[args.batch:args.batch+viznum], masks[args.batch:args.batch+viznum],
                                                    recon_imgs[args.batch:args.batch+viznum], recon_masks[args.batch:args.batch+viznum],
                                                    imgs[args.batch*2:args.batch*2+viznum], masks[args.batch*2:args.batch*2+viznum],
                                                    trans_masks[0:viznum], trans_masks[args.batch:args.batch+viznum]), dim=0), 128).cpu()
            utils.save_image(
                sample,
                f"log/%s/%05d.jpg"%(args.task, (idx+1)),
                nrow=viznum*2,
                normalize=True,
                range=(-1, 1),
            )

            #plt.figure(figsize=(10,10), dpi=120)
            #visualize(torchvision.utils.make_grid(sample, viznum*2, 2))
            #plt.show()    
            
            
if __name__ == "__main__":

    parser = TrainOptions()
    args = parser.parse()
    print('*'*98)
    
    if not os.path.exists("log/%s/"%(args.task)):
        os.makedirs("log/%s/"%(args.task))
    
    device = 'cuda'
    netAE = AutoEncoder().to(device)
    netAE.init_weights('kaiming', 0.02)
    optimizer_AE = torch.optim.Adam(netAE.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    print('Create models successfully!')
    
    ufiles = os.listdir(args.unpaired_data_root) 
    natural_sort(ufiles)
    udataset_sizes = [600] * len(ufiles)
    udataset_sizes[-1] = 29000 # for faces
    ulabels = list(range(len(ufiles)))

    pfiles = os.listdir(args.paired_data_root) 
    natural_sort(pfiles)
    pdataset_sizes = [600] * len(pfiles)
    plabels = list(range(len(pfiles)))    
    
    # for unpaired data
    udataloader = create_imagemasklabel_dataloader(args.unpaired_data_root, args.unpaired_mask_root, 
                                                   ufiles, udataset_sizes, ulabels)

    # for paired data
    pdataloader = create_imagemasklabel_dataloader(args.paired_data_root, args.paired_mask_root, 
                                                   pfiles, pdataset_sizes, plabels, pair=True)

    print('Create dataloaders successfully!')
    
    vgg_loss = VGGLoss()
    vgg_loss.vgg = vgg_loss.vgg.to(device)

    train(args, udataloader, pdataloader, netAE, optimizer_AE, vgg_loss, device)