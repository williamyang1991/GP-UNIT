import tempfile
import numpy as np
import torch
from util import save_image
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from cog import BasePredictor, Path, Input

from model.generator import Generator
from model.content_encoder import ContentEncoder
from model.sampler import ICPTrainer


TASKS = [
    "female2male",
    "male2female",
    "cat2dog",
    "dog2cat",
    "face2cat",
    "cat2face",
    "bird2dog",
    "dog2bird",
    "bird2car",
    "car2bird",
]


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda"
        self.netEC = ContentEncoder()
        self.netEC.eval()
        self.netG = Generator()
        self.netG.eval()
        self.sampler = ICPTrainer(np.empty([0, 256]), 128)

    def predict(
        self,
        task: str = Input(
            choices=TASKS,
            default="cat2dog",
            description="Choose style type.",
        ),
        content: Path = Input(
            description="Input content image, it will be resized to 256",
        ),
        style: Path = Input(
            default=None,
            description="Input style image, it will be resized to 256",
        ),
    ) -> Path:

        self.netEC.load_state_dict(
            torch.load(
                "checkpoint/content_encoder.pt",
                map_location=lambda storage, loc: storage,
            )
        )
        ckpt = torch.load(
            f"checkpoint/{task}.pt", map_location=lambda storage, loc: storage
        )
        self.netG.load_state_dict(ckpt["g_ema"])
        self.sampler.icp.netT.load_state_dict(ckpt["sampler"])

        self.netEC = self.netEC.to(self.device)
        self.netG = self.netG.to(self.device)
        self.sampler.icp.netT = self.sampler.icp.netT.to(self.device)
        print("Model successfully loaded!")

        Ix = F.interpolate(
            load_image(str(content)), size=256, mode="bilinear", align_corners=True
        )
        if style is not None:
            Iy = F.interpolate(
                load_image(str(style)), size=256, mode="bilinear", align_corners=True
            )

        #seed = 233
        #torch.manual_seed(seed)
        with torch.no_grad():
            content_feature = self.netEC(Ix.to(self.device), get_feature=True)
            
        with torch.no_grad():
            if style is not None:
                I_yhat, _ = self.netG(content_feature, Iy.to(self.device))
            else:
                style_features = self.sampler.icp.netT(torch.randn(4, 128).to(self.device))
                I_yhat, _ = self.netG(content_feature.repeat(4,1,1,1), style_features, useZ=True)
                I_yhat = torchvision.utils.make_grid(I_yhat, 2, 0)

        out_path = Path(tempfile.mkdtemp()) / "output.png"
        save_image(I_yhat[0].cpu(), str(out_path))
        return out_path


def load_image(filename):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img = Image.open(filename).convert("RGB")
    img = transform(img)
    return img.unsqueeze(dim=0)
