#ifndef _OOBLECK_HETERO_PLANNING_PIPELINE_TEMPLATE_H_
#define _OOBLECK_HETERO_PLANNING_PIPELINE_TEMPLATE_H_

#include "execution_result.h"
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

#define DEBUG_PIPELINE_TEMPLATE

namespace oobleck {

struct SingleNodeSpec {
  std::string node_type; // 'A100', 'H100', 'B100' etc.
  int num_gpus;
  double compute_power;
  SingleNodeSpec(std::string node_type, int num_gpus, double compute_power)
      : node_type(node_type), num_gpus(num_gpus), compute_power(compute_power) {
  }
};

extern std::vector<SingleNodeSpec> node_specs;
extern std::map<std::string, int> node_specs_map;

struct NodeConfig {
  int node_type_idx;
  int num_nodes;
  int num_gpus;
  int num_total_gpus;
  double compute_power;
  // memory
  NodeConfig(std::string node_type, int num_nodes, int num_gpus_per_node,
             double compute_power)
      : num_nodes(num_nodes), num_gpus(num_gpus_per_node),
        compute_power(compute_power) {
    if (node_specs_map.find(node_type) == node_specs_map.end()) {
      node_specs.push_back(
          SingleNodeSpec(node_type, num_gpus_per_node, compute_power));
      node_specs_map[node_type] = node_specs.size() - 1;
      node_type_idx = node_specs.size() - 1;
    } else {
      node_type_idx = node_specs_map[node_type];
    }
    num_total_gpus = num_nodes * num_gpus_per_node;
  }

  std::string to_string() const {
    return node_specs[node_type_idx].node_type + "[" +
           std::to_string(num_nodes) +
           "nodes:" + std::to_string(num_total_gpus) + 
           "gpus per node:" + std::to_string(num_gpus) + "]";
  }

  bool operator==(const NodeConfig &other) const {
    return node_type_idx == other.node_type_idx && num_nodes == other.num_nodes;
  }
};

struct HeteroNodeSpec {
  std::vector<NodeConfig> node_specs;
  int num_total_nodes = 0;
  int num_total_gpus = 0;
  int idx_to_only_node = -1; // index to the node that has only one node; other
                             // nodes are 0.
  HeteroNodeSpec(const std::vector<NodeConfig> &node_specs)
      : node_specs(node_specs), num_total_nodes(0) {
    update_fields();
  }
  HeteroNodeSpec(const std::vector<NodeConfig> &node_specs, int num_total_nodes)
      : node_specs(node_specs), num_total_nodes(num_total_nodes) {}
  HeteroNodeSpec() : num_total_nodes(0), idx_to_only_node(-1) {}

  const std::vector<NodeConfig> &get() const { return node_specs; }

  int size() const { return num_total_nodes; }

  const NodeConfig &at(const int idx) const { return node_specs[idx]; }

  // update num_total_nodes and idx_to_only_node when node_specs is updated
  void update_fields() {
    num_total_nodes = 0;
    for (int i = 0; i < node_specs.size(); i++) {
      auto &node_spec = node_specs[i];
      num_total_nodes += node_spec.num_nodes;
      num_total_gpus += node_spec.num_total_gpus;

      // point idx_to_only_node to the only node
      if (node_spec.num_nodes == 1)
        idx_to_only_node = i;
    }

    // if there is more than one node, idx_to_only_node is not valid
    if (num_total_nodes != 1) {
      idx_to_only_node = -1;
    }
  }

  // subtract another HeteroNodeSpec(a subset) from this
  HeteroNodeSpec subtract(const HeteroNodeSpec &other,
                          bool need_update = true) const {
    std::vector<NodeConfig> new_node_specs = node_specs;
    int total_nodes = num_total_nodes;
    for (int i = 0; i < node_specs.size(); i++) {
      new_node_specs[i].num_nodes -= other.node_specs[i].num_nodes;
      total_nodes -= other.node_specs[i].num_nodes;
      assert(new_node_specs[i].num_nodes >= 0 && total_nodes >= 0);
    }
    if (need_update) {
      return HeteroNodeSpec(new_node_specs);
    } else {
      return HeteroNodeSpec(new_node_specs, total_nodes);
    }
  }

  std::string to_string() const {
    std::string result = "[";
    result += "node#: " + std::to_string(num_total_nodes) + ", ";
    result += "gpu#: " + std::to_string(num_total_gpus) + ", ";
    result += "idx: " + std::to_string(idx_to_only_node) + ", ";
    for (auto &config : node_specs) {
      result += "[" + config.to_string() + "] ";
    }
    result += "]";
    return result;
  }

  std::string get_cache_key() const;
  std::string get_cache_key_recovery() const;
};

class HeteroPipelineTemplate {
public:
  HeteroPipelineTemplate() = default;
  HeteroPipelineTemplate(
      const std::vector<std::shared_ptr<StageExecutionResult>>
          &stage_execution_results,
      const double t1, const double t2, const double t3, const double kstar_lat,
      const double iteration_time, const int num_mbatches, int num_layers,
      const HeteroNodeSpec &node_spec)
      : stage_execution_results_(stage_execution_results), t1_(t1), t2_(t2),
        t3_(t3), kstar_lat_(kstar_lat), iteration_time_(iteration_time),
        num_mbatches_(num_mbatches), node_spec_(node_spec) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all for each node type
    // 2. stages cover all layers
    std::vector<int> num_gpus_used(node_spec.node_specs.size(), 0);
    for (auto &stage : stage_execution_results_) {
      std::cout << stage->to_string() << std::endl;
      num_gpus_used[stage->node_type_idx_] += stage->num_gpus_;
    }

    for (int i = 0; i < num_gpus_used.size(); i++) {
      PRINT("num_gpus_used for node type " + std::to_string(i) + " : " +
            std::to_string(num_gpus_used[i]));
      PRINT("availiavble num_gpus for node type " + std::to_string(i) + " : " +
            std::to_string(
                node_spec.node_specs[i].num_nodes *
                node_spec.node_specs[i]
                    .num_gpus));
      assert(num_gpus_used[i] <=
             node_spec.node_specs[i].num_nodes *
                 node_spec.node_specs[i].num_gpus);
    }

    int stage_num_layers = 0;
    for (auto &stage : stage_execution_results_) {
      stage_num_layers += stage->num_layers();
    }
    assert(stage_num_layers == num_layers);
  }

  const double get_t1() const { return t1_; }
  const double get_t2() const { return t2_; }
  const double get_t3() const { return t3_; }
  const double get_kstar_latency() const { return kstar_lat_; }
  const double get_iteration_time() const { return iteration_time_; }
  const int get_num_mbatches() const { return num_mbatches_; }
  std::string to_string() const {
    std::string repr = "<oobleck.HeteroPipelineTemplate.[";
    repr += "t: " + std::to_string(get_iteration_time()) + ", ";
    repr += "t1: " + std::to_string(get_t1()) + ", ";
    repr += "t2: " + std::to_string(get_t2()) + ", ";
    repr += "t3: " + std::to_string(get_t3()) + ", ";
    repr += "kstar_latency: " + std::to_string(get_kstar_latency()) + ", ";
    repr += "num_mbatches: " + std::to_string(get_num_mbatches()) + ", ";
    repr += "stages: [" + '\n';
    for (const auto &stage : get_stages()) {
      repr += stage->to_string() + ", " + '\n';
    }
    repr.pop_back();
    repr.pop_back();
    repr += "], ";
    repr += "node_spec: " + get_node_spec().to_string();
    repr += "]>";
    return repr;
  }

  const std::vector<std::shared_ptr<StageExecutionResult>> &get_stages() const {
    return stage_execution_results_;
  }

  const HeteroNodeSpec &get_node_spec() const { return node_spec_; }

private:
  std::vector<std::shared_ptr<StageExecutionResult>> stage_execution_results_;
  const double t1_;
  const double t2_;
  const double t3_;
  const double kstar_lat_;
  const double iteration_time_;
  const int num_mbatches_;
  HeteroNodeSpec node_spec_;
};

} // namespace oobleck

#endif