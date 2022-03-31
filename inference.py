import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from model.generator import Generator
from model.content_encoder import ContentEncoder
from model.sampler import ICPTrainer
from util import load_image, save_image


class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Inference of GP-UNIT")
        self.parser.add_argument("--content", type=str, default='./data/afhq/images512x512/test/dog/flickr_dog_000572.jpg', help="path to the content image")
        self.parser.add_argument("--style", type=str, default=None, help="path to the style image, if not specified using randomly sampled styles")
        self.parser.add_argument("--batch", type=int, default=6, help="number of randomly sampled styles")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path to save the output images")
        self.parser.add_argument("--name", type=str, default='translation', help="filename to save the generated images")
        self.parser.add_argument("--generator_path", type=str, default='./checkpoint/dog2cat.pt', help="path to the saved generator")
        self.parser.add_argument("--content_encoder_path", type=str, default='./checkpoint/content_encoder.pt', help="path to the saved content encoder")
        self.parser.add_argument("--device", type=str, default='cuda', help="`cuda` for using GPU and `cpu` for using CPU")
        
    def parse(self):
        self.opt = self.parser.parse_args()        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

if __name__ == "__main__":

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    device = args.device
    netEC = ContentEncoder()
    netEC.eval()
    netG = Generator()
    netG.eval()
    sampler = ICPTrainer(np.empty([0,256]), 128)
    
    netEC.load_state_dict(torch.load(args.content_encoder_path, map_location=lambda storage, loc: storage))
    ckpt = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(ckpt['g_ema'])
    sampler.icp.netT.load_state_dict(ckpt['sampler'])
    
    netEC = netEC.to(device)
    netG = netG.to(device)
    sampler.icp.netT = sampler.icp.netT.to(device)

    print('Load models successfully!')
    
    if args.style is None:
        print('Perform latent-guided translation to generate %d images'%(args.batch))
        save_name = args.name+'_%s'%(os.path.basename(args.content).split('.')[0])
    else:
        print('Perform exemplar-guided translation with the style image %s'%(os.path.basename(args.style)))
        save_name = args.name+'_%s_to_%s'%(os.path.basename(args.content).split('.')[0], os.path.basename(args.style).split('.')[0])
    
    with torch.no_grad():
        viz = []
        # load content image and comuput content features
        Ix = F.interpolate(load_image(args.content), size=256, mode='bilinear', align_corners=True)
        content_feature = netEC(Ix.to(device), get_feature=True)
        
        # perform translation
        if args.style is not None:
            Iy = F.interpolate(load_image(args.style), size=256, mode='bilinear', align_corners=True)
            I_yhat, _ = netG(content_feature, Iy.to(device))
        else:
            style_features = sampler.icp.netT(torch.randn(args.batch, 128).to(device))
            I_yhat, _ = netG(content_feature, style_features, useZ=True)

    print('Generate images successfully!')
    
    if args.style is not None:
        save_image(I_yhat[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))
        save_image(torchvision.utils.make_grid(torch.cat([Ix, Iy, I_yhat.cpu()], dim=0), 3, 2), 
               os.path.join(args.output_path, save_name+'_overview.jpg'))
    else:
        for i in range(args.batch):
            save_image(I_yhat[i].cpu(), os.path.join(args.output_path, save_name+'_%d'%(i)+'.jpg'))
        save_image(torchvision.utils.make_grid(torch.cat([Ix, I_yhat.cpu()], dim=0), args.batch+1, 2), 
               os.path.join(args.output_path, save_name+'_overview.jpg'))

    print('Save images successfully!')