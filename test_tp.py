import torch

from my_gpt import FeedForward
from tp import _apply_tp_ffn, maybe_init_dist


if __name__ == "__main__":
    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        torch.cuda.set_device(rank)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    torch.manual_seed(1234)
    ffn = FeedForward(intermediate_size=512, dim=192)
    _apply_tp_ffn(ffn)
    print(ffn.w1.weight.shape)
    print(ffn.w3.weight.shape)
    print(ffn.w2.weight.shape)
    ffn.cuda()
    x = torch.randn(2, 192).to("cuda")
    y = ffn(x)
    print(y[0, :10])
    print(y.shape)