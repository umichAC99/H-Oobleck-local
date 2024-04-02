#include "pipeline_recovery.h"

// build up dp_choices_ and avail_devices_ and initial dp_ for this problem
void ButtomUpDPipelineRecoverSolver::preprocess(
    const PipelineTemplate &longest_pipeline,
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {

  assert(longest_pipeline != nullptr && "longest_pipeline is nullptr");
  auto current_stages = longest_pipeline->get_stages();

  // Initialize avail_devices_
  for (int i = 0; i < hetero_node_spec_.size(); i++) {
    avail_devices_[i] = hetero_node_spec_.node_specs[i].num_nodes *
                        hetero_node_spec_.node_specs[i].num_gpus;
  }

  // Initialize dp size
  dp_.resize(current_stages.size() * 2);

  // Initialize all possible dp_choices_ for different node types and dp
  // starting state
  /* default result is that we merge $covered_stages stages starting from 0 with
   * $j devices in type i */
  /* DP[covered_stage] = (assigned_device, execution_result) */
  for (int i = 0; i < hetero_node_spec_.size(); i++) {
    dp_choices_[i] = std::vector<Choice>();
    DeviceResource assigned_device(hetero_node_spec_.size(), 0);
    for (int j = 1; j <= hetero_node_spec_.node_specs[i].num_gpus; j *= 2) {
      int covered_stage = (int)floor(scaling_factors_[i] * j);
      dp_choices_[i].push_back(Choice(j, covered_stage));
      assigned_device[i] = j;

      auto execution_result = merge_stages(current_stages, 0, covered_stage - 1,
                                           j, i, layer_execution_results[i]);

      dp_[covered_stage] = DPState(assigned_device, execution_result);
    }
  }
}