from nlp.models.embedding import PositionalEmbedding
from nlp.models.gpt2 import GTP2Model
from nlp.models.masking import PadFutureMask
import torch
from nlp.models.encoder import TextEncoder
from nlp.attention.attention import MultiHeadAttention


def test_gpt2_general():
    model = GTP2Model(vocab_size=1000, d_model=64, n_layers=2, heads=8).eval()
    past = None

    # loss = model(x, layer_past=past,labels=x)
    # print(loss)
    for i in range(2):
        print(i)
        x, past = model(torch.randint(82, (2, 3)), layer_past=past)

    for p in past:
        assert p[0].shape == (2, 6, 64)


def test_gpt2_train():
    model = GTP2Model(vocab_size=1000, d_model=64, n_layers=2, heads=8).eval()
    past = None
    x = torch.randint(82, (2, 3))
    loss = model(x, layer_past=past, labels=x)
    print(loss.item())


def test_positional_embedding():
    a = PositionalEmbedding(1024, 10)
    input_ids = torch.randint(1000, (1, 1))
    print(input_ids)

    b = a(input_ids, 1)
    assert b.shape == (1, 1, 10)


def test_masking():
    mask = PadFutureMask(idx=0, future=True)
    x = torch.randint(50, (2, 5))
    key = torch.rand((2, 5, 64))
    y = mask(x, key)
    assert y.shape == (2, 5, 10)
    z = mask(x)
    assert z.shape == (2, 5, 5)


def test_image():
    from torchvision import models
    mobile_v3 = models.mobilenet_v3_small(pretrained=True)


def test_text_encoder():
    x = torch.randint(0, 100, (3, 7))
    # encoder = EncoderLayer(units=120, d_model=768, heads=8)
    encoder = TextEncoder(vocab_size=99, num_layers=4, num_heads=8, units=128, d_model=768, max_seq=7)
    a = encoder(x)
    assert a.shape == (3, 7, 768)


def test_multi_head_att():
    layer = MultiHeadAttention(heads=2)
    q = torch.zeros((10, 16))
    k = torch.zeros((20, 16))
    v = torch.zeros((20, 32))
    out = layer(q, k, v)
    print(out.shape == (10, 32))
    q = torch.zeros((4, 10, 16))
    k = torch.zeros((4, 20, 16))
    v = torch.zeros((4, 20, 32))
    mask = torch.zeros((4, 10, 20)).bool()

    print(layer(q, k, v, mask).shape)


if __name__ == '__main__':
    # test_gpt2_general()
    # test_gpt2_train()
    # test_positional_embedding()
    # test_masking()
    test_multi_head_att()
