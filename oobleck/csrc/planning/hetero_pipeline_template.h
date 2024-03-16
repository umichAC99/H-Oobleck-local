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
  char node_type; // 'A', 'B', 'C' etc.
  int num_nodes;
  int num_gpus_per_node;
  double compute_power;
};

struct HeteroNodeSpec {
  std::vector<SingleNodeSpec> node_specs;
};

class HeteroPipelineTemplate {
public:
  typedef std::vector<std::vector<std::shared_ptr<StageExecutionResult>>>
      HeteroStageExecutionResults;
  HeteroPipelineTemplate(
      const HeteroStageExecutionResults &stage_execution_results,
      int num_layers, const HeteroNodeSpec &node_spec)
      : stage_execution_results_(stage_execution_results),
        node_spec_(node_spec) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all # GPUs used by stage results
    // 2. stages cover all layers
    int num_gpus_used = 0;
    for (int i = 0; i < stage_execution_results_.size(); i++) {
      auto &stages = stage_execution_results_[i];
      for (auto &stage : stages) {
        std::cout << stage->to_string() << std::endl;
        num_gpus_used += stage->num_gpus_;
      }
      assert(num_gpus_used == node_spec_.node_specs[i].num_nodes *
                                  node_spec_.node_specs[i].num_gpus_per_node);

      int stage_num_layers = 0;
      for (auto &stage : stages) {
        stage_num_layers += stage->num_layers();
      }
    }
  }

  const HeteroStageExecutionResults &get_stages() const {
    return stage_execution_results_;
  }

  const HeteroNodeSpec &get_node_spec() const { return node_spec_; }

private:
  HeteroStageExecutionResults stage_execution_results_;
  HeteroNodeSpec node_spec_;
};

} // namespace oobleck

#endif