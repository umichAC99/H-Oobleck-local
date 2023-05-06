#ifndef _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_
#define _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_

#include <oneapi/tbb/concurrent_hash_map.h>
#include <atomic>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "execution_result.h"

namespace oobleck {

class PipelineTemplate {
 public:
  PipelineTemplate(
      const std::vector<StageExecutionResult>& stage_execution_results,
      const int num_layers,
      const int num_nodes,
      const int num_gpus_per_node)
      : stage_execution_results_(stage_execution_results),
        num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all # GPUs used by stage results
    // 2. stages cover all layers
    int num_gpus_used = 0;
    for (auto& stage : stage_execution_results_) {
      std::cout << stage.to_string() << std::endl;
      num_gpus_used += stage.device_num();
    }
    assert(num_gpus_used == num_nodes * num_gpus_per_node);

    int stage_num_layers = 0;
    for (auto& stage : stage_execution_results_) {
      stage_num_layers += stage.num_layers();
    }
    assert(stage_num_layers == num_layers);
  }

  std::vector<StageExecutionResult> get_stage_execution_results() const {
    return stage_execution_results_;
  }
  int get_num_nodes() const { return num_nodes_; }
  int get_num_gpus_per_node() const { return num_gpus_per_node_; }

 private:
  std::vector<StageExecutionResult> stage_execution_results_;
  const int num_nodes_;
  const int num_gpus_per_node_;
};

class PipelineTemplateGenerator {
 public:
  static CacheMap dc_cache_;
  static cppcoro::static_thread_pool thread_pool_;

  std::vector<PipelineTemplate> create_pipeline_templates(
      const std::string& model_name,
      const std::string& model_tag,
      const int microbatch_size,
      const std::tuple<int, int> num_nodes,
      const int num_gpus_per_node);

 private:
  std::shared_ptr<std::vector<LayerExecutionResult>> get_profiler_results(
      const std::string& model_name,
      const std::string& model_tag,
      const int microbatch_size);

  cppcoro::task<std::shared_ptr<DCExecutionResult>> divide_and_conquer(
      std::shared_ptr<std::vector<LayerExecutionResult>>
          layer_execution_results,
      const std::tuple<int, int>& layer_indices,
      const int num_stages,
      const int num_nodes,
      const int num_gpus_per_node);

  std::atomic<unsigned long> cache_hit_;
  std::atomic<unsigned long> cache_miss_;
};

}  // namespace oobleck

#endif