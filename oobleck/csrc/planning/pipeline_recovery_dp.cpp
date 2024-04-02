#include "pipeline_recovery.h"
namespace oobleck {
// pretty print all resources, choices and dp states
void ButtomUpDPPipelineRecoverSolver::print() {
  // Step1. print all available devices
  std::cout << "Available Devices: " << std::endl;
  for (int i = 0; i < avail_devices_.size(); i++) {
    std::cout << "Node Type " << i << ": " << avail_devices_[i] << '\t';
  }
  std::cout << std::endl;

  // Step2. print all possible choices
  std::cout << "All Possible Choices: " << std::endl;
  std::cout << "[NOTE]: (x, y) means we use x devices to cover y stages"
            << std::endl;
  for (int i = 0; i < dp_choices_.size(); i++) {
    std::cout << "Node Type " << i << ": ";
    for (int j = 0; j < dp_choices_[i].size(); j++) {
      std::cout << "(" << dp_choices_[i][j].first << " : "
                << dp_choices_[i][j].second << " stages)" << '\t';
    }
    std::cout << std::endl;
  }

  // Step3. print all dp states
  std::cout << "DP States: " << std::endl;
  for (int i = 0; i < dp_.size(); i++) {
    std::cout << "DP[" << i << "]: ";
    if (dp_[i].second == nullptr) {
      std::cout << "infeasible" << std::endl;
      continue;
    }
    std::cout << "Assigned Devices: ";
    for (int j = 0; j < dp_[i].first.size(); j++) {
      std::cout << dp_[i].first[j] << '\t';
    }
    std::cout << "Execution Result: " << dp_[i].second->to_string()
              << std::endl;
  }
}

// build up dp_choices_ and avail_devices_ and initial dp_ for this problem
void ButtomUpDPPipelineRecoverSolver::preprocess(
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {

  assert(longest_pipeline_ != nullptr && "longest_pipeline_ is nullptr");
  PRINT("start preprocess!");
  auto current_stages = longest_pipeline_->get_stages();

  // Initialize avail_devices_
  for (int i = 0; i < hetero_node_spec_.size(); i++) {
    avail_devices_[i] = hetero_node_spec_.node_specs[i].num_nodes *
                        hetero_node_spec_.node_specs[i].num_gpus;
  }

  // Initialize dp size
  dp_.resize(current_stages.size() * 2,
             {DeviceResource(hetero_node_spec_.size(), 0), nullptr});

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

      // merge stages from 0 to covered_stage - 1
      auto execution_result = merge_stages(current_stages, 0, covered_stage - 1,
                                           j, i, layer_execution_results[i]);
      // dp_[covered_stage] = DPState(assigned_device, execution_result);
      update_dp_slot(covered_stage, execution_result, assigned_device);
      //   PRINT("dp_" + std::to_string(covered_stage) + " is updated with " +
      //   execution_result->to_string());
    } // for j
  }   // for i
}

HeteroPipelineTemplate ButtomUpDPPipelineRecoverSolver::solve(
    const std::vector<PipelineTemplate> &pipeline_templates,
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {
  assert(pipeline_templates.size() > 0 && "pipeline_templates is empty");
  assert(layer_execution_results.size() == hetero_node_spec_.size() &&
         "layer_execution_results size is not equal to hetero_node_spec_ size");

  // preprocess initial dp states and choices
  longest_pipeline_ = &pipeline_templates[pipeline_templates.size() - 1];
  preprocess(layer_execution_results);

  // pprint all initial states when debug
  DEBUG_STMT(print());

  return HeteroPipelineTemplate(
      longest_pipeline_->get_stages(), longest_pipeline_->get_t1(),
      longest_pipeline_->get_t2(), longest_pipeline_->get_t3(),
      longest_pipeline_->get_kstar_latency(),
      longest_pipeline_->get_iteration_time(), num_mbatches_,
      layer_execution_results[0]->size(), hetero_node_spec_);
}
} // namespace oobleck