class CLIPConfig:
    def __init__(self,
                 img_dim=256,
                 img_blocks=6,
                 img_heads=8,
                 dim=128,
                 img_dim_head=128,
                 img_dim_linear_block=1024,
                 drop_pro=0,
                 in_channels=3,
                 patch_dim=16,
                 vocab_size=50000,
                 layers=4,
                 units=512,
                 text_d_model=768,
                 text_heads=8,
                 text_max_seq=128,
                 text_drop=0.1,
                 classification=False,
                 num_classes=0,
                 img_text_dim=768,
                 T=-1
                 ):
        # img config
        self.img_dim = img_dim
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.num_classes = num_classes
        self.dim = dim
        self.img_blocks = img_blocks
        self.img_heads = img_heads
        self.img_dim_linear_block = img_dim_linear_block
        self.img_dim_head = img_dim_head
        self.drop_pro = drop_pro
        self.classification = classification
        # text config
        self.vocab_size = vocab_size
        self.layers = layers
        self.units = units
        self.text_d_model = text_d_model
        self.text_heads = text_heads
        self.text_max_seq = text_max_seq
        self.text_drop = text_drop

        # out_dim
        self.img_text_dim = img_text_dim
        # t - learned temperature parameter
        self.T = T
