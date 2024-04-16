import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import os

from oobleck.module.sharding import shard_model #might have to fix this

from graph2hlo import graph2hlo #comes from the graph2hlo file

# simple for loop to iterate
def run_all_conversions(model, test_input):
    # Shard the model into graph modules
    graph_modules = shard_model(model, ['input'], ['split1', 'split2'])  # can play with these values

    results = []
    for layer in enumerate(self.model.layers):
        hlo_string = graph2hlo(layer, test_input)
        results.append(hlo_string)
    return results

# Define a simple neural network as an example - CHATGPT stuff
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(10, 5)  # Example layer

    def forward(self, x):
        return torch.relu(self.linear(x))

# Create an instance of the model and a dummy input
model = SimpleNN()
dummy_input = torch.randn(1, 10)  # Adjust the shape according to your model's requirements

# Run the conversion process
hlo_strings = run_all_conversions(model, dummy_input)


# Define the directory to store the output HLO files
output_directory = 'hlo_outputs'
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

for i, hlo in enumerate(hlo_strings):
    file_path = os.path.join(output_directory, f'hlo_output_{i}.txt')
    with open(file_path, 'w') as file:
        file.write(hlo)
    print(f'HLO output saved to {file_path}')