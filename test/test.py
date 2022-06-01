from nlp.models.embedding import PositionalEmbedding
from nlp.models.gpt2 import GTP2Model
from nlp.models.masking import PadFutureMask
import torch


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


if __name__ == '__main__':
    # test_gpt2_general()
    # test_gpt2_train()
    # test_positional_embedding()
    test_masking()
