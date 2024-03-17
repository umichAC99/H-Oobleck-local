#ifndef _OOBLECK_HETERO_PLANNING_PIPELINE_TEMPLATE_H_
#define _OOBLECK_HETERO_PLANNING_PIPELINE_TEMPLATE_H_

#include "execution_result.h"
#include <atomic>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <pybind11/pybind11.h>
#include <string>
#include <tuple>
#include <vector>

namespace oobleck {

struct SingleNodeSpec {
  std::string node_type;  // 'A100', 'H100', 'B100' etc.
  int num_gpus;
  double compute_power;
  SingleNodeSpec(std::string node_type, int num_gpus, double compute_power)
      : node_type(node_type), num_gpus(num_gpus), compute_power(compute_power) {}
};

extern std::vector<SingleNodeSpec> node_specs;

struct NodeConfig {
  int node_type_idx;
  int num_nodes;
  // memory
  NodeConfig(std::string node_type, int num_nodes, int num_gpus_per_node, double compute_power)
      : num_nodes(num_nodes){
        node_specs.push_back(SingleNodeSpec(node_type, num_gpus_per_node, compute_power));
        node_type_idx = node_specs.size() - 1;
      }
};

struct HeteroNodeSpec {
  std::vector<NodeConfig> node_specs;
  int num_total_nodes;
  int idx_to_only_node; // index to the node that has only one node; other nodes are 0.
  HeteroNodeSpec(std::vector<NodeConfig> node_specs) : node_specs(node_specs), num_total_nodes(0) {
    for (auto& node_spec : node_specs) {
      num_total_nodes += node_spec.num_nodes;
    }
    idx_to_only_node = -1;
  }
  HeteroNodeSpec(): num_total_nodes(0), idx_to_only_node(-1) {}
};

class HeteroPipelineTemplate {
public:
  HeteroPipelineTemplate() = default;
  HeteroPipelineTemplate(
      const std::vector<std::shared_ptr<StageExecutionResult>>&
                       stage_execution_results,
      int num_layers, const HeteroNodeSpec &node_spec)
      : stage_execution_results_(stage_execution_results),
        node_spec_(node_spec) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all for each node type
    // 2. stages cover all layers
    std::vector<int> num_gpus_used(node_spec.node_specs.size(), 0);
    for (auto& stage : stage_execution_results_) {
      std::cout << stage->to_string() << std::endl;
      num_gpus_used[stage->node_type_idx_] += stage->num_gpus_;
    }

    for (int i = 0; i < num_gpus_used.size(); i++) {
      assert(num_gpus_used[i] == node_spec.node_specs[i].num_nodes * node_spec.node_specs[i].num_gpus_per_node);
    }

    int stage_num_layers = 0;
    for (auto& stage : stage_execution_results_) {
      stage_num_layers += stage->num_layers();
    }
    assert(stage_num_layers == num_layers);
    
  }

  const std::vector<std::shared_ptr<StageExecutionResult>> &get_stages() const {
    return stage_execution_results_;
  }

  const HeteroNodeSpec &get_node_spec() const { return node_spec_; }

private:
  std::vector<std::shared_ptr<StageExecutionResult>> stage_execution_results_;
  HeteroNodeSpec node_spec_;
};

} // namespace oobleck

#endif