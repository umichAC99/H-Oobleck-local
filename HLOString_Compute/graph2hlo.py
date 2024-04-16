import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import tensorflow as tf
import os
from copy import deepcopy 

def graph2hlo(graphModule, sampleInput): #removed the test input cause we need it on the real layers
    '''
    input a grahModule, output the HLO string for the module
    @param graphModule: a pytorch.fx.GraphModule (the layer of Oobleck)
    @param inputWidth: width of the first layer in the graphModule
    @param nSamples: default 1000. Batch size/ number of samples to run through
    @return String: the hlostring
    '''
    #1) create model.onnx
    # Specify the path to the ONNX file
    onnx_file_path = "model.onnx"
    #dummy_input = (torch.randn(1, 1024), torch.randn(1, 1024), torch.randn(1, 1024))  # fixed this hopefully
    #TODO need to find this size dynamically by extracting from the graphModule
    testInput = sampleInput
    #graphModule.forward = funct
    #graphModuleCopy = deepcopy(graphModule)
    ##funct = lambda input_ids, attention_mask, labels : torch.tensor(attention_mask)
    #graphModuleCopy.bad = graphModule
    #def funct(input_ids, attention_mask, labels):
    #    #torch.tensor(graphModule.forward(input_ids, attention_mask, labels))
    #    return torch.tensor(attention_mask)
    #funct = lambda input_ids, attention_mask, labels : torch.tensor(graphModule.forward(input_ids, attention_mask, labels)[1])
    #graphModuleCopy.forward = funct
    #print(graphModuleCopy)

    import types
    import types

    def copy_func(f, name=None):
        '''
        return a function with same code, globals, defaults, closure,
        and 
        name (or provide a new name)
    '''
        fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)
        fn.__dict__.update(f.__dict__) 
        return fn
    #print(graphModule)
    tmpF = copy_func(graphModule.forward)
    def funct(*args):
        result = tmpF(graphModule, *args)
        return torch.tensor(result[0]) if isinstance(result, tuple) else torch.tensor(result)
    graphModule.forward = funct
    #testInput = torch.randn(3, sampleInput.shape[1])

    #ZKN THIS WORKS!!!!!!!!!!!!!!
    #funct = lambda input_ids, attention_mask, labels : torch.tensor(attention_mask)
    #graphModule.forward = funct
    torch.onnx.export(graphModule,               # model being run
                      testInput,               # model input (or a tuple for multiple inputs)
                      onnx_file_path,            # where to save the model
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=17,          # the ONNX version to export the model to (note 17 is highest)
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
    hloString = xla_fn.experimental_get_compiler_ir(testInput[0])(stage="hlo")

    return hloString

def convertAllLayers(layers, sampleInputs):
    """
    @param layers: iterable of torch.fx.graphmodule (as found in Oobleck
      self.model.layers
    @param widths: iterable of corresponding input widths
    Returns a string delimited by the DELIMITER local variable, separating hlo
    strings for each layer.
    @postcondition: writes the string to a file hloOut.txt
    """
    allStrings = ""
    DELIMITER = "\nzkn\n"
    for layer, sampleInput in zip(layers, sampleInputs):
        hloString = graph2hlo(layer, sampleInput)
        allStrings += hloString
        print(hloString)
        allStrings += DELIMITER
    allStrings = allStrings[:-81] #trim the final delimiter
    with open("hloOut.txt", 'w') as file:
        file.write(allStrings)
    return allStrings

def main():
    '''
    This is a simple main function demonstrates it works for a very basic neural
    network layer
    '''
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
    simple_nn2 = SimpleNN()
    # Use torch.fx.symbolic_trace to
    # create a GraphModule
    graph_module = symbolic_trace(simple_nn)
    graph_module2 = symbolic_trace(simple_nn)
    graph_modules = [graph_module, graph_module2]
    widths = [10, 10]

    
    out = convertAllLayers(graph_modules, widths)
    #out = graph2hlo(graph_module)
    with open('out.txt', 'w') as file:
       file.write(out)

if __name__ == "__main__":
    main()
