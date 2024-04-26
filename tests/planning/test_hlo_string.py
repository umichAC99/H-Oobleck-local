import pytest

from tests.conftest import OobleckSingleProcessTestCase
import HLOString_Compute.graph2hlo as g2hlo
import torch
import copy
from torch_mlir import torchscript

class TestOobleckHLOString(OobleckSingleProcessTestCase):
    def generateInputs(self, model):
        prev_in = tuple(
            [
                t.detach().clone().to('cpu')  # Ensure tensors are moved to CPU
                for t in model.sample_inputs.values()
            ]
        )
        inputs = []
        for layer in model.layers:
                    # Check each parameter in the layer
            for name, param in layer.named_parameters():
                # Handle hierarchical parameter names
                submodules = name.split('.')[:-1]  # Split and remove the last element which is the param name
                param_name = name.split('.')[-1]   # Get the actual parameter name
                sublayer = layer
                for submodule in submodules:
                    sublayer = getattr(sublayer, submodule)  # Navigate to the correct submodule
                
                if param.device.type == 'meta':
                    # Create a new tensor with the same properties but on 'cpu'
                    real_data = torch.randn_like(param, device='cpu', dtype=param.dtype)
                    # Replace the parameter with the new tensor
                    setattr(sublayer, param_name, torch.nn.Parameter(real_data))
            cpu_layer = layer.to("cpu")  # Move each layer to CPU
            inputs.append(prev_in)
            output = cpu_layer(*prev_in)
            #print(output)
            if isinstance(output, tuple):
                next_in = tuple(
                    [
                        t.detach().clone().to("cpu")
                        if isinstance(t, torch.Tensor) else t
                        for t in output
                    ]
                )
            elif isinstance(output, torch.Tensor):
                next_in = output.detach().clone().to("cpu")
            prev_in = next_in
        return inputs

    def test_hloStringTest(self):
        model = self.factory.get_bert_model()

        sample_inputs = self.generateInputs(model)
        
        for i, layer in enumerate(model.layers):
            print(layer)
            out_stablehlo_mlir_path = f"results/layer_{i}.mlir"
            module = torchscript.compile(layer, sample_inputs[i], output_type=torchscript.OutputType.STABLEHLO)
            with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
                outf.write(str(module))