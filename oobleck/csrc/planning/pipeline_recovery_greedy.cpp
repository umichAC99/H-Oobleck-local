#include "pipeline_recovery.h"
#include <cmath>
namespace oobleck {

/*
    @brief:
        A greedy algorithm to recover(assign) heterogenous nodes to a homogenous
   pipeline template
    @args:
        const PipelineTemplate& pipeline_template;
        const std::vector<float>& scaling_factors_;
        const HeteroNodeSpec& hetero_node_spec_;
        const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results;
        const CacheMap* dc_cache_;

    @return:
        HeteroPipelineTemplate

    @assumption:
        scaling_factors_, hetero_node_spec_ and layer_execution_results are
   sorted based on scaling_factors_ in ascending order (first one is the
   weakest node)

    @pseduocode:
        for i : 1...len(node_spec):
            used_device = 0
            total_device = node_spec[i].num_nodes * node_spec[i].num_gpus
            while (used_device < total_device):
                stages = pipeline_template.get_stages()
                min_time = INF
                min_idx = -1
                for j : 1...len(stages):
                    assigned_device = stages[j].device / scaling_factors_[i]
                    assigned_device = ceil(assigned_device)
                    min_time = try_assign(j, assigned_device, profile)
                    if min_time < min_time:
                        min_time = min_time
                        min_idx = j
                assign(min_idx, assigned_device, profile)
                used_device += assigned_device

*/

void empty_node_spec(HeteroNodeSpec &spec);

void replace_device(HeteroNodeSpec &spec, int src, int dest, int src_device,
                    int dest_device);

std::shared_ptr<HeteroPipelineTemplate> GreedyPipelineRecoverSolver::solve_one(
    const PipelineTemplate &pipeline_template,
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {

  dc_cache_.clear();
  assert(pipeline_template.get_num_layers() ==
             layer_execution_results[0]->size() &&
         "Layer Execution Results size is not equal to pipeline template size");
  auto curr_stages = pipeline_template.get_stages();
  HeteroNodeSpec curr_spec, left_spec, right_spec;

  // update dc cache for current stage
  update_homo_dc_cache(curr_stages);

  // initialize curr_spec to be homogenous cluster
  curr_spec.node_specs = hetero_node_spec_.node_specs;
  for (int i = 0; i < curr_spec.node_specs.size(); i++) {
    if (i == 0) {
      curr_spec.node_specs[i].num_nodes = pipeline_template.get_num_nodes();
      curr_spec.node_specs[i].num_gpus =
          pipeline_template.get_num_gpus_per_node();
      curr_spec.node_specs[i].num_total_gpus =
          pipeline_template.get_num_gpus_per_node() *
          pipeline_template.get_num_nodes();
    } else {
      curr_spec.node_specs[i].num_nodes = 0;
      curr_spec.node_specs[i].num_gpus =
          pipeline_template.get_num_gpus_per_node();
      curr_spec.node_specs[i].num_total_gpus = 0;
    }
  }
  curr_spec.update_fields();
  // PRINT("Curr Spec: " + curr_spec.to_string());
  // PRINT("Scaling Fact: ");
  for (int i = 0; i < scaling_factors_.size(); i++) {
    PRINT(std::to_string(scaling_factors_[i]) + " ");
  }

  // start greedy algorithm
  std::shared_ptr<oobleck::DCExecutionResult> min_cost_dc_result = nullptr;
  for (int i = hetero_node_spec_.node_specs.size() - 1; i > 0; i--) {
    int used_device = 0;
    int total_device = hetero_node_spec_.node_specs[i].num_nodes *
                       hetero_node_spec_.node_specs[i].num_gpus;
    while (used_device < total_device) {
      double min_time = std::numeric_limits<double>::max();
      int min_idx = -1;
      int min_time_assigned_device = -1;
      int assigned_device = -1;
      min_cost_dc_result = nullptr;

      // update left and right ptrs, empty left first
      left_spec = curr_spec;
      right_spec = curr_spec;
      empty_node_spec(left_spec);
      for (int j = 0; j < curr_stages.size(); j++) {

        // assign device to stage based on scaling factor f
        double assigned_device_f =
            curr_stages[j]->num_gpus_ / scaling_factors_[i];
        // PRINT("Assigned Device F: " + std::to_string(assigned_device_f) +
        //       "Scaling Factor: " + std::to_string(scaling_factors_[i]));
        if (assigned_device_f + used_device > total_device)
          assigned_device_f = total_device - used_device;

        assigned_device = ceil(assigned_device_f);
        assert(assigned_device > 0 && "Assigned device is not set");

        // try to assign node idx i with assigned_device to stage, update left
        // and right spec
        int curr_stage_gpu = curr_stages[j]->num_gpus_;
        int curr_stage_node_idx = curr_stages[j]->node_type_idx_;

        // remove current device from right spec
        replace_device(right_spec, curr_stage_node_idx, i, curr_stage_gpu, 0);

        // only try assign if current stage is assigned to the weakest node
        if (curr_stage_node_idx == 0 && assigned_device_f > 0.6f) {
          auto dc_result =
              try_assign(j, i, assigned_device, layer_execution_results[i],
                         curr_stages, left_spec, right_spec);
          if (dc_result->get_t() < min_time) {
            min_time = dc_result->get_t();
            min_idx = j;
            min_time_assigned_device = assigned_device;
            min_cost_dc_result = dc_result;
          }
        } // if

        // add current device to left spec
        replace_device(left_spec, i, curr_stage_node_idx, 0, curr_stage_gpu);
      } // for

      // if no stage is assigned, throw exception
      if (min_idx == -1) {
        std::string curr_stages_str;
        for (int i = 0; i < curr_stages.size(); i++) {
          curr_stages_str += curr_stages[i]->to_string() + '\n';
        }
        throw RecoveryFailException(
            "Failed to recover pipeline when assigning node type: " +
            std::to_string(i) + '\n' + "curr_stages: " + curr_stages_str +
            '\n');
      }

      // PRINT("[RESULT]: Min Time: " + std::to_string(min_time) + '\n' +
      //       " Min Idx: " + std::to_string(min_idx) + '\n' +
      //       " Assigned Device: " + std::to_string(min_time_assigned_device) +
      //       '\n' + " Used Device: " + std::to_string(used_device) + '\n' +
      //       " Total Device: " + std::to_string(total_device) + '\n' +
      //       " Current Spec: " + curr_spec.to_string() + '\n' +
      //       " Min Cost DC Result: " + min_cost_dc_result->to_string());
      // update current spec and current stages
      replace_device(curr_spec, 0, i, curr_stages[min_idx]->num_gpus_,
                     min_time_assigned_device);
      curr_stages = min_cost_dc_result->get_stages();

      // update dc cache with current result
      update_dc_cache(min_idx, curr_stages, left_spec, right_spec);
      used_device += min_time_assigned_device;
    } // while
  }

  return std::make_shared<HeteroPipelineTemplate>(
      curr_stages, min_cost_dc_result->get_t1(), min_cost_dc_result->get_t2(),
      min_cost_dc_result->get_t3(), min_cost_dc_result->get_kstar_latency(),
      min_cost_dc_result->get_t(), num_mbatches_,
      layer_execution_results[0]->size(), hetero_node_spec_);
}

HeteroPipelineTemplate GreedyPipelineRecoverSolver::solve(
    const std::vector<PipelineTemplate> &pipeline_templates,
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {
  std::shared_ptr<HeteroPipelineTemplate> best_result = nullptr;
  double best_t = std::numeric_limits<double>::max();
  for (int i = 0; i < pipeline_templates.size(); i++) {
    try {
      PRINT("Solving pipeline template: " + std::to_string(i) + " with " +
            std::to_string(pipeline_templates[i].get_stages().size()) +
            " stages");
      PRINT("Original pipeline template: " + pipeline_templates[i].to_string());
      auto result = solve_one(pipeline_templates[i], layer_execution_results);
      PRINT("Recovered pipeline template: " + result->to_string());
      if (result->get_iteration_time() < best_t) {
        best_t = result->get_iteration_time();
        best_result = result;
      }
    } catch (RecoveryFailException &e) {
      PRINT(e.reason);
    }
  }
  assert(best_result != nullptr && "Best result is not found");
  return *best_result;
}
} // namespace oobleck