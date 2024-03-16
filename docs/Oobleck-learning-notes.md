# Oobleck Learning Notes

The main gpu-workload mapping entry is at [`line:83 create_pipeline_templates` at pipeline_template.cpp](../oobleck/csrc/planning/pipeline_template.cpp): 

* It takes three inputs: 1. `std::shared_ptr<LayerExecutionResults> layer_execution_results` the profiling result for each layer; 2. `const std::tuple<int, int>& num_nodes` the nodes range (for example, if num_nodes = (1, 3), then this function will generate templates for 1, 2, and 3 nodes). For our project, we should just use range (x, x); 3. `const std::tuple<int, int>& num_gpus` the number of gpus per node.
* Then, for each template (template that has 1, 2, 3, ... nodes), it will call [`divide_and_conquer(layers, num_stages, num_nodes, num_gpus_per_node)`](../oobleck/csrc/planning/pipeline_template.cpp) and find the `num_stages` that minimizes the total execution time.

Here are some todos for us to extend Oobleck in supporting heterogenous nodes:
1. Oobleck is using the [`profiler`](../oobleck/planning/profiler.py) to get the profiling result for each layer. This profiler will store the result in `/tmp/oobleck/profiles/{model_name}-{tag}/{layers|allreduce_in_node|allreduce_across_nodes}`; then,  [`get_profile_results`](../oobleck/csrc/planning/pipeline_template.cpp) will read the profiling result from the file. What we need to do is simply run our cost model and store the results to the same directory(Need one profiling result per node type). This is sufficient for the profiling part.

2. We need to modify `divide_and_conquer` to support taking fractional number of nodes. For example, in the base case where we have one stage but a fractional number of nodes (0.7)
```cpp
  // Base case (conquer phase)
  if (num_stages == 1) {
    assert(num_nodes == 1); // TODO: We can remove this assert
    // If there is only one stage, assign all layers to that stage TODO: put fractional number of nodes in the StageExecutionResult constructor and scale up the execution time
    auto stage = std::make_shared<StageExecutionResult>(
        layer_execution_results, layer_indices, num_gpus_per_node);
    auto result = std::make_shared<DCExecutionResult>(stage, num_nodes,
                                                      num_gpus_per_node);
    dc_cache_.insert({key, result});
    // accessor->second = result;
    co_return result;
  }
```

3. ~~We need to write an API in [`pipeline_template.cpp`](../oobleck/csrc/planning/pipeline_template.cpp) that implements our ideas, the sketch code will be like:~~ [This is legacy since it's not optimal. The main goal is to effectively find the optimal solution]
```python
    # nodes is a directory, key is the node type, value is a tuple [num_nodes, num_gpus_per_node]
    AC99(node_types, total_nodes):
        node_types = sort(node_types, key=lambda x: x.performance)
        template = ();
        allocated_layers = set();
        for m in node_types:
            # scaling heterogenous nodes to homogenous nodes
            num_nodes = hardware_scaling(total_nodes);
            layer_execution_results = get_profile_results(m);

            # It will run create_pipeline_templates as discussed above but ignore the allocated_layers
            result = create_pipeline_templates(layer_execution_results, num_nodes, allocated_layers);

            # 1. pick the template that minimize the total execution time for nodes[m] nodes
            # 2. record the allocated layers
            process_template(template, result, m, total_nodes, allocated_layers);

            # remove node m from nodes
            remove(nodes, m);
        return template;

```

4. We need to write a test case for our algorithm in [`test_pipeline_template.py`](../tests/planning/test_pipeline_template.py). 
  To run pytest, do `pytest -s tests/planning/test_pipeline_template.py::test_create_pipeline_templates_maxnode` in the root directory of the project. You don't need to recompile the python test code every time.

5. ~~We also need to discuss the optimality of our algorithm in the documentation.~~

5. We need to extend current oobleck gpu mapping technique such that it can find the optimal solution. Our approach will try to prune the search space but will still try to find the solution closed the optimal one.
