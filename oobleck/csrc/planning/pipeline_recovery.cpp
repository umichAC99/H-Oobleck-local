#include "pipeline_recovery.h"
#include <cmath>
namespace oobleck {

/*
    @brief:
        A greedy algorithm to recover(assign) heterogenous nodes to a homogenous
   pipeline template
    @args:
        const PipelineTemplate& pipeline_template_;
        const std::vector<float>& scaling_factors_;
        const HeteroNodeSpec& hetero_node_spec_;
        const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results_;
        const CacheMap* dc_cache_;

    @return:
        HeteroPipelineTemplate

    @assumption:
        scaling_factors_, hetero_node_spec_ and layer_execution_results_ are
   sorted based on scaling_factors_ in ascending order (first one is the
   weakest node)

    @pseduocode:
        for i : 1...len(node_spec):
            used_device = 0
            total_device = node_spec[i].num_nodes * node_spec[i].num_gpus
            while (used_device < total_device):
                stages = pipeline_template_.get_stages()
                min_time = INF
                min_idx = -1
                for j : 1...len(stages):
                    assigned_device = stages[j].device / scaling_factors_[i]
                    if assigned_device < 0.5
                        continue
                    else
                        assigned_device = ceil(assigned_device)
                    min_time = try_assign(j, assigned_device, profile)
                    if min_time < min_time:
                        min_time = min_time
                        min_idx = j
                assign(min_idx, assigned_device, profile)
                used_device += assigned_device

*/

static DCExecutionResult::key get_dc_key(int num_stages, int start_layer_idx,
                                          int end_layer_idx,
                                          const HeteroNodeSpec& spec) {
    bool is_homo = spec.num_total_nodes == spec.node_specs[0].num_nodes;
    std::string device_key;
    if (is_homo){
        device_key =  DCExecutionResult::get_device_indices_key(
                          spec.node_specs[0].num_nodes, spec.node_specs[0].num_gpus, 0);
    } else{
        device_key = spec.get_cache_key_recovery();
    }
    return std::make_tuple(num_stages, start_layer_idx, end_layer_idx, device_key);
}

// try to assign node idx i with assigned_device to stage, update left and right spec
static void update_node_spec(std::shared_ptr<StageExecutionResult> stage, int node_idx, int assigned_device, HeteroNodeSpec& left_spec, HeteroNodeSpec& right_spec){
    assert(stage->node_type_idx_ == 0 && "Trying to assign a stage that has been assigned");
}

std::shared_ptr<oobleck::DCExecutionResult>
BasePipelineRecoverSolver::try_assign(
    int idx, int node_type, int assigned_device,
    const std::shared_ptr<LayerExecutionResults> &profile, HeteroNodeSpec& curr,
    std::vector<std::shared_ptr<StageExecutionResult>> & stages,
    const HeteroNodeSpec& left, const HeteroNodeSpec& right) const {

    // find DCExecutionResult from 0...idx-1
    std::shared_ptr<oobleck::DCExecutionResult> left_result = nullptr;
    if (idx > 0){
        auto key = get_dc_key(idx, 0, idx-1, left);
        auto it = dc_cache_->find(key);
        if (it != dc_cache_->end()){
            left_result = it->second;
        }
    }

    // find DCExecutionResult from idx+1...end
    std::shared_ptr<oobleck::DCExecutionResult> right_result = nullptr;
    if (idx < stages.size()-1){
        auto key = get_dc_key(stages.size()-idx-1, idx+1, stages.size()-1, right);
        auto it = dc_cache_->find(key);
        if (it != dc_cache_->end()){
            right_result = it->second;
        }
    }

    auto new_stage = std::make_shared<StageExecutionResult>(
        profile, std::make_tuple(stages[idx]->layer_indices_[0], stages[idx]->layer_indices_[1]), assigned_device, node_type);
  return nullptr;
}

static void empty_node_spec(HeteroNodeSpec& spec){
    for (int i = 0; i < spec.node_specs.size(); i++) {
        spec.node_specs[i].num_nodes = 0;
        spec.node_specs[i].num_gpus = 0;
        spec.node_specs[i].num_total_gpus = 0;
    }
    spec.update_fields();
}

HeteroPipelineTemplate GreedyPipelineRecoverSolver::solve() const {

  assert(dc_cache_ != nullptr && "DC Cache is not set");

  auto curr_stages = pipeline_template_.get_stages();
  HeteroNodeSpec curr_spec, left_spec, right_spec;

  // initialize curr_spec to be homogenous cluster
  curr_spec.node_specs = hetero_node_spec_.node_specs;
  for (int i = 0; i < curr_spec.node_specs.size(); i++) {
    if (i == 0){
        curr_spec.node_specs[i].num_nodes = pipeline_template_.get_num_nodes();
        curr_spec.node_specs[i].num_gpus = pipeline_template_.get_num_gpus_per_node();
        curr_spec.node_specs[i].num_total_gpus = pipeline_template_.get_num_gpus_per_node() * pipeline_template_.get_num_nodes();
    }   else {
        curr_spec.node_specs[i].num_nodes = 0;
        curr_spec.node_specs[i].num_gpus = pipeline_template_.get_num_gpus_per_node();
        curr_spec.node_specs[i].num_total_gpus = 0;
    }
  }
  curr_spec.update_fields();
  PRINT("Curr Spec: " + curr_spec.to_string());

  assert(false && "Not implemented");
  // start greedy algorithm
  for (int i = hetero_node_spec_.node_specs.size()-1; i > 0; i++) {
    int used_device = 0;
    int total_device = hetero_node_spec_.node_specs[i].num_nodes *
                       hetero_node_spec_.node_specs[i].num_gpus;
    while (used_device < total_device) {
      double min_time = std::numeric_limits<double>::max();
      int min_idx = -1;
      int min_time_assigned_device = -1;
      int assigned_device = -1;
      for (int j = 0; j < curr_stages.size(); j++) {

        // update left and right ptrs, empty left first
        left_spec = curr_spec;
        right_spec = curr_spec;
        empty_node_spec(left_spec);

        // assign device to stage based on scaling factor f
        double assigned_device_f = curr_stages[j]->num_gpus_ / scaling_factors_[i];
        if (assigned_device_f < 0.5)
          continue;
        else if (assigned_device_f + used_device > total_device)
          assigned_device_f = total_device - used_device;
        
        assigned_device = ceil(assigned_device_f);
        assert(assigned_device > 0 && "Assigned device is not set");

        // try to assign node idx i with assigned_device to stage, update left and right spec

        // TODO: instead of -1, we should minus the number of gpus
        curr_spec.node_specs[0].num_total_gpus -= curr_stages[j]->num_gpus_;
        curr_spec.node_specs[i].num_total_gpus += assigned_device;
        right_spec.node_specs[0].num_total_gpus -= curr_stages[j]->num_gpus_;
        auto dc_result = try_assign(j, i, assigned_device, layer_execution_results_[i], curr_spec, curr_stages, left_spec, right_spec);
        if (dc_result->get_t() < min_time) {
          min_time = dc_result->get_t();
          min_idx = j;
          min_time_assigned_device = assigned_device;
        }
        left_spec.node_specs[0].num_total_gpus += curr_stages[j]->num_gpus_;
      }
      // assign(min_idx, assigned_device, profile);
      assert(false && "Not Implemented!");
      assert(min_time_assigned_device != -1 && "Assigned device is not set");
      used_device += min_time_assigned_device;
    }
  }

  return HeteroPipelineTemplate(
      pipeline_template_.get_stages(), pipeline_template_.get_t1(),
      pipeline_template_.get_t2(), pipeline_template_.get_t3(),
      pipeline_template_.get_kstar_latency(),
      pipeline_template_.get_iteration_time(),
      pipeline_template_.get_num_mbatches(),
      pipeline_template_.get_num_layers(), hetero_node_spec_);
}
} // namespace oobleck