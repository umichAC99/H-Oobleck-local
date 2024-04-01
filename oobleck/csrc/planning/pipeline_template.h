#ifndef _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_
#define _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_

#include "execution_result.h"
#include "hetero_pipeline_template.h"
#include "oobleck_utils.h"
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
class PipelineTemplate {
public:
  PipelineTemplate(const std::vector<std::shared_ptr<StageExecutionResult>>
                       &stage_execution_results,
                   const double t1, const double t2, const double t3,
                   const double kstar_lat, const double iteration_time,
                   const int num_mbatches, const int num_layers,
                   const int num_nodes, const int num_gpus_per_node)
      : stage_execution_results_(stage_execution_results), t1_(t1), t2_(t2),
        t3_(t3), kstar_lat_(kstar_lat), num_mbatches_(num_mbatches),
        num_layers_(num_layers), iteration_time_(iteration_time),
        num_nodes_(num_nodes), num_gpus_per_node_(num_gpus_per_node) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all # GPUs used by stage results
    // 2. stages cover all layers
    int num_gpus_used = 0;
    for (auto &stage : stage_execution_results_) {
      std::cout << stage->to_string() << std::endl;
      num_gpus_used += stage->num_gpus_;
    }
    assert(num_gpus_used == num_nodes * num_gpus_per_node);

    int stage_num_layers = 0;
    for (auto &stage : stage_execution_results_) {
      stage_num_layers += stage->num_layers();
    }
    assert(stage_num_layers == num_layers);
  }

  std::string to_string() const {
    std::string repr = "<oobleck.HomoPipelineTemplate.[";
    repr += "t: " + std::to_string(get_iteration_time()) + ", ";
    repr += "t1: " + std::to_string(get_t1()) + ", ";
    repr += "t2: " + std::to_string(get_t2()) + ", ";
    repr += "t3: " + std::to_string(get_t3()) + ", ";
    repr += "kstar_latency: " + std::to_string(get_kstar_latency()) + ", ";
    repr += "num_mbatches: " + std::to_string(get_num_mbatches()) + ", ";
    repr += "stages: [" + '\n';
    for (const auto &stage : get_stages()) {
      repr += stage->to_string() + "," + '\n';
    }
    repr.pop_back();
    repr.pop_back();
    repr += "], ";
    repr += "]>";
    return repr;
  }

  const double get_t1() const { return t1_; }
  const double get_t2() const { return t2_; }
  const double get_t3() const { return t3_; }
  const double get_kstar_latency() const { return kstar_lat_; }
  const double get_iteration_time() const { return iteration_time_; }
  const int get_num_layers() const { return num_layers_; }
  const int get_num_mbatches() const { return num_mbatches_; }
  const std::vector<std::shared_ptr<StageExecutionResult>> &get_stages() const {
    return stage_execution_results_;
  }
  int get_num_nodes() const { return num_nodes_; }
  int get_num_gpus_per_node() const { return num_gpus_per_node_; }

  std::map<int, std::vector<int>> get_rank_grid(std::vector<int> ranks) {
    // Return a map of layer index to a list of ranks
    std::map<int, std::vector<int>> rank_grid;
    for (auto &stage : stage_execution_results_) {
      std::vector<int> stage_ranks(ranks.begin(),
                                   ranks.begin() + stage->num_gpus_);
      ranks.erase(ranks.begin(), ranks.begin() + stage->num_gpus_);

      std::vector<int> layer_ranks(num_gpus_per_node_);
      auto it = layer_ranks.begin();

      // If length of `stage_ranks` is less than num_gpus_per_node,
      // adjust it so that it has the same length as num_gpus_per_node
      const int repeat_count = num_gpus_per_node_ / stage->num_gpus_;
      for (const int rank : stage_ranks) {
        std::fill_n(it, repeat_count, rank);
        std::advance(it, repeat_count);
      }

      // push per-layer ranks to the result
      for (const int layer_index : stage->layer_indices_) {
        rank_grid[layer_index] = layer_ranks;
      }
    }

    assert(ranks.size() == 0);
    return std::move(rank_grid);
  }

private:
  std::vector<std::shared_ptr<StageExecutionResult>> stage_execution_results_;
  const double t1_;
  const double t2_;
  const double t3_;
  const double kstar_lat_;
  const double iteration_time_;
  const int num_mbatches_;
  const int num_layers_;
  const int num_nodes_;
  const int num_gpus_per_node_;
};

std::shared_ptr<LayerExecutionResults>
get_profile_results(const std::string &model_name, const std::string &model_tag,
                    const int microbatch_size,
                    const std::string &node_type = "");

// get profile results for each node type
std::vector<std::shared_ptr<LayerExecutionResults>>
get_hetero_profile_results(const std::vector<std::string> &model_names,
                           const std::vector<std::string> &model_tags,
                           const int microbatch_size,
                           const std::vector<std::string> &node_types);

class PipelineTemplateGenerator {
public:
  CacheMap dc_cache_;
  cppcoro::static_thread_pool thread_pool_ = cppcoro::static_thread_pool(1);

  CacheMap *get_dc_cache() { return &dc_cache_; }

  void print_dc_cache() const {
    PRINT("DC CACHE:");
    for (const auto &entry : dc_cache_) {
      std::string string_key = std::to_string(std::get<0>(entry.first)) + "[" +
                               std::to_string(std::get<1>(entry.first)) + "-" +
                               std::to_string(std::get<2>(entry.first)) + "]" +
                               std::get<3>(entry.first);
      if (entry.second != nullptr)
        PRINT("key: " + string_key + " -> " +
              "value: " + entry.second->to_string());
    }
  }

  std::vector<PipelineTemplate> create_pipeline_templates(
      std::shared_ptr<LayerExecutionResults> layer_execution_results,
      const std::tuple<int, int> &num_nodes, const int num_gpus_per_node,
      const int num_mbatches = 0);

  // create one hetero pipeline template based on node spec and layer execution
  // results
  HeteroPipelineTemplate create_hetero_pipeline_template(
      std::vector<std::shared_ptr<LayerExecutionResults>>
          layer_execution_results,
      const HeteroNodeSpec &node_spec, const int num_mbatches = 0);

private:
  cppcoro::task<std::shared_ptr<DCExecutionResult>> divide_and_conquer(
      std::shared_ptr<LayerExecutionResults> layer_execution_results,
      const std::tuple<int, int> layer_indices, const int num_stages,
      const int num_nodes, const int num_gpus_per_node, const int num_mbatches);

  cppcoro::task<std::shared_ptr<DCExecutionResult>>
  divide_and_conquer(const std::vector<std::shared_ptr<LayerExecutionResults>>
                         &layer_execution_results,
                     const std::tuple<int, int> layer_indices,
                     const int num_stages, const HeteroNodeSpec &node_spec,
                     const int num_mbatches);

  std::atomic<unsigned long> cache_hit_;
  std::atomic<unsigned long> cache_miss_;
};

} // namespace oobleck

#endif