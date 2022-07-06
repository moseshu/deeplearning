import numpy as np
import torch
import torch.nn as nn
from vitransformer import ViT
from nlp.models.encoder import TextEncoder
from config import CLIPConfig
from loss import clip_loss

"""

# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer

# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
"""


class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig):
        super(CLIP, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.image_encoder = ViT(
            img_dim=config.img_dim,
            in_channels=config.in_channels,
            patch_dim=config.patch_dim,
            num_classes=config.num_classes,
            dim=config.dim,
            blocks=config.img_blocks,
            heads=config.img_heads,
            dim_linear_block=config.img_dim_linear_block,
            dim_head=config.img_dim_head,
            dropout=config.drop_pro,
            classification=config.classification
        )

        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            num_layers=config.layers,
            units=config.units,
            d_model=config.text_d_model,
            num_heads=config.text_heads,
            max_seq=config.text_max_seq,
            dropout=config.text_drop
        )
        tokens = (config.img_dim // config.patch_dim) ** 2
        self.W_I = nn.Parameter(torch.randn(tokens * config.dim, config.img_text_dim))
        self.W_T = nn.Parameter(torch.randn(config.text_d_model * config.text_max_seq, config.img_text_dim))
        self.norm_l1 = nn.LayerNorm(config.img_text_dim)
        self.norm_l2 = nn.LayerNorm(config.img_text_dim)

    def forward(self, image: torch.Tensor, text: torch.Tensor, text_mask=None, image_task=None):
        # image.shape = [bs,c,h,w]  text.shape = [bs,max_seq]
        image_encode = self.image_encoder(image, image_task)  # [bs,patchs, dim]
        text_encode = self.text_encoder(text, text_mask)  # [bs,max_seq,text_d_model]

        image_out = image_encode[:, 0, :]
        text_out = text_encode[:, 0, :]

        I_E = torch.einsum("bi,ij->bj", [image_out, self.W_I])
        T_E = torch.einsum("bi,ij->bj", [text_out, self.W_T])
        image_E = self.norm_l1(I_E)
        text_E = self.norm_l2(T_E)

        logit_scale = self.logit_scale.exp()

        logits_per_text = torch.einsum("bi,ik->bk", [text_E, image_E]) * logit_scale

        logits_per_image = logits_per_text.T
        loss = clip_loss(logits_per_text)
        out_put = {"logits_per_text": logits_per_text,
                   "logits_per_image": logits_per_image,
                   "loss": loss,
                   "vision_model_output": image_out,
                   "text_model_output": text_out
                   }
        return out_put


if __name__ == '__main__':
    config = CLIPConfig()

    model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=None, dim=512, classification=False)
    x = torch.rand(2, 3, 256, 256)
    y = model(x)  # [2,10]
    print(y.shape)  # batch, classes

    a = torch.tensor([[1, 2, -1], [2, 4, -1]])
    b = torch.tensor([[2, 1, -1], [4, 3, -1]])
    c = torch.einsum("bi,ik->bk", [a, b.T])
    print(c)
    print(np.exp(1))
    print(torch.exp(torch.tensor([1])))
