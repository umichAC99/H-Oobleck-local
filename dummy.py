import torch

GpuType = str  # 'A100', 'V100'
LayerExecutionResult = dict  # { 'forward': float, 'backward': float, 'mem_required': (float, float), ... }
Layer = 'LayerIdentifier' 

# We have TODO this part: function to initialize layers, model, and the profiler
initialize_profiler, model_layers = ... 

# Profiling results go here with real time hardware profiling data
gpu_profile_data: dict[GpuType, list[LayerExecutionResult]] = {
    'A100': [...],
    'V100': [...],
}

def get_cost_model(gpu_types: list[GpuType], model_layers: list[Layer]) -> list[dict]:
    profiler = initialize_profiler() 
    cost_model = []

    for gpu_type in gpu_types:
        execution_results = gpu_profile_data[gpu_type]
        gpu_cost_info = []

        # Iterate for layer
        for layer in model_layers:
            layer_profile = execution_results[layer.index]
            
            # Calculat costs
            forward_time = layer_profile['forward']
            backward_time = layer_profile['backward']
            mem_required = layer_profile['mem_required']

            # communication cost - need to recode this
            comm_cost = 0.1 * forward_time 

            gpu_cost_info.append({
                'forward_time': forward_time,
                'backward_time': backward_time,
                'memory': mem_required,
                'communication_cost': comm_cost
            })

        cost_model.append({
            'gpu_type': gpu_type,
            'layers': gpu_cost_info
        })

    return cost_model
