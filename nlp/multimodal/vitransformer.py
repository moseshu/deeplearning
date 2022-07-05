import torch
import torch.nn as nn
from img_encoder import TransformerEncoder


class ViT(nn.Module):
    def __init__(self,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0,
                 classification=True
                 ):
        super(ViT, self).__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible'
        self.p = patch_dim,
        self.classification = classification
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                              dim_head=self.dim_head,
                                              dim_linear_block=dim_linear_block,
                                              dropout=dropout)

        if self.classification:
            self.CLS_TOKEN = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
            self.out = nn.Linear(dim, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img: torch.Tensor, mask=None):
        assert img.dim() == 4
        batch_size, channels, H, W, = img.shape  # [bs.c,h,w]
        patch_x = H / self.p
        patch_y = W / self.p

        img_patches = img.contiguous().view((batch_size, patch_x * patch_y, channels * (self.p ** 2)))
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)
        if self.classification:
            img_patches = torch.cat(
                (self.expand_cls_to_batch(batch_size), img_patches), dim=1)

        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)
        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)  # [bs,tokens or tokens+1,dim]

        if self.classification:
            # we index only the cls token for classification.
            return self.out(y[:, 0, :])
        else:
            return y
