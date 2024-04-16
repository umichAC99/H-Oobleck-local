import torch
#import torch.nn as nn
from torch.fx import symbolic_trace
import tensorflow as tf
import os

def graph2hlo(graphModule): #removed the test input cause we need it on the real layers
    #1) create model.onnx
    # Specify the path to the ONNX file
    onnx_file_path = "model.onnx"
    dummy_input = (torch.randn(1, 1024), torch.randn(1, 1024), torch.randn(1, 1024))  # fixed this hopefully
    torch.onnx.export(graphModule,               # model being run
                      dummy_input,               # model input (or a tuple for multiple inputs)
                      onnx_file_path,            # where to save the model
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],     # the model's input names
                      output_names=['output'],   # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes if applicable
                                    'output': {0: 'batch_size'}}
                     )

    #2) convert model.onnx to saved_model keras. Requires docker
    os.system("onnx2tf -i model.onnx -okv3")

    #3) load saved model into tensorflow
    model = tf.keras.models.load_model("saved_model/model_float32_v3.keras")
    xla_fn = tf.function(model, jit_compile=True) # compiles in xla

    #4) convert to hloString
    hloString = xla_fn.experimental_get_compiler_ir(testInput)(stage="hlo")

    return hloString

# Define a simple neural network module
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(10, 5)  # An example layer

    def forward(self, x):
        x = self.linear(x)
        return torch.relu(x)

# Create an instance of the
# module
simple_nn = SimpleNN()
# Use torch.fx.symbolic_trace to
# create a GraphModule
graph_module = symbolic_trace(simple_nn)

# Create a dummy input that matches the input shape the model expects
dummy_input = torch.randn(1, 10)

out = graph2hlo(simple_nn, dummy_input)
with open('out.txt', 'w') as file:
   file.write(out)
