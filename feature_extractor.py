# Wrapper to extract features from intermediate layers
from typing import Dict, Iterable, Callable
import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: {'input': torch.empty(0), 'output': torch.empty(0)} for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self._features[layer_id]['input'] = input[0]
            self._features[layer_id]['output'] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.model(x)
        return output, self._features