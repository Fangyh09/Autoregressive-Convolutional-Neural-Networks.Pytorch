import torch
from model import EncoderCNN

if __name__ == "__main__":
    data = torch.randn((10, 1, 5, 20, 20))
    # off_model = OffsetCNN()
    # sig_model = SignificanceCNN()
    enc_model = EncoderCNN()
    out = enc_model(data)
    print("out.shape", out.shape)

