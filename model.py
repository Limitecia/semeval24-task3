
import torch.nn as nn

class EmotionCausalMdoel(nn.Module):
    def __init__(
        self,
        pretrained: str,
        embed_size: int
    ):