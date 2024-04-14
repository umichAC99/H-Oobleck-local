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
  for (int i = 0; i < hetero_node_spec_.node_specs.size(); i++) {
    avail_devices_[i] = hetero_node_spec_.node_specs[i].num_nodes *
                        hetero_node_spec_.node_specs[i].num_gpus;
  }

  // Initialize dp size
  dp_.resize(current_stages.size() + 1,
             {DeviceResource(hetero_node_spec_.node_specs.size(), 0), nullptr});

  // Initialize all possible dp_choices_ for different node types and dp
  // starting state
  /* default result is that we merge $covered_stages stages starting from 0 with
   * $j devices in type i */
  /* DP[covered_stage] = (assigned_device, execution_result) */
  for (int i = 0; i < hetero_node_spec_.node_specs.size(); i++) {
    dp_choices_[i] = std::vector<Choice>();
    DeviceResource assigned_device(hetero_node_spec_.node_specs.size(), 0);
    for (int j = 1; j <= hetero_node_spec_.node_specs[i].num_gpus; j *= 2) {
      int covered_stage = (int)scaling_factors_[i] * j;
      dp_choices_[i].push_back(Choice(j, covered_stage));
      assigned_device[i] = j;

      // merge stages from 0 to covered_stage - 1
      auto execution_result = merge_stages(current_stages, 0, covered_stage - 1,
                                           j, i, layer_execution_results[i]);
      // dp_[covered_stage] = DPState(assigned_device, execution_result);
      update_dp_slot(covered_stage, execution_result, assigned_device);
      //   PRINT("dp_" + std::to_string(covered_stage) + " is updated with " +
      // PRINT("dp_" + std::to_string(covered_stage) + " is updated with " +
      //       execution_result->to_string());
      // DEBUG_STMT(print());
    } // for j
  }   // for i

}

/*

for i in (2...S):
  for choice in dp_choices_:
    dp[i] = 
      min(dp[i], merge_results(
          dp[i - choice.stages], 
          result([i - choice.stages...i], choice.devices
          )
        )
      )

*/
HeteroPipelineTemplate ButtomUpDPPipelineRecoverSolver::solve(
    const std::vector<PipelineTemplate> &pipeline_templates,
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {
  PRINT("layer_execution_results size is " + std::to_string(layer_execution_results.size()));
  PRINT("hetero_node_spec_ is " + hetero_node_spec_.to_string());
  assert(pipeline_templates.size() > 0 && "pipeline_templates is empty");
  assert(layer_execution_results.size() == hetero_node_spec_.node_specs.size() &&
         "layer_execution_results size is not equal to hetero_node_spec_ size");
  // preprocess initial dp states and choices
  longest_pipeline_ = &pipeline_templates[pipeline_templates.size() - 1];
  preprocess(layer_execution_results);

  // pprint all initial states when debug
  DEBUG_STMT(print());

  // Start DP
  auto curr_stages = longest_pipeline_->get_stages();
  for (int i = 1; i < dp_.size(); i++) {
    for (int node_type_idx = 0; node_type_idx < dp_choices_.size(); node_type_idx++){
      for (const auto& choice : dp_choices_[node_type_idx]) {
        std::cout << "In DP[" << i << "] " << std::endl;
        std::cout << "Choice: " << choice.first << " devices to cover " << choice.second << " stages" << std::endl;
        // if we cannot cover $choice.second stages from i, continue
        if (i - choice.second < 0) {
          continue;
        }
        DPState prev_state = dp_[i - choice.second];
        // prev_state is infeasible, continue
        if (prev_state.second == nullptr) {
          continue;
        }
        DeviceResource assigned_device = prev_state.first;
        // if apply this choice will exceed the device limit, continue
        if (over_device_limit(assigned_device, node_type_idx, choice.first)) {
          continue;
        }
        assigned_device[node_type_idx] += choice.first;
        auto execution_result = std::make_shared<DCExecutionResult>(
          prev_state.second,
          merge_stages(
            curr_stages, i - choice.second, i - 1, choice.first, node_type_idx,
            layer_execution_results[node_type_idx]),
          num_mbatches_);
        std::cout << "DP[" << i - choice.second <<"] + " << choice.first << " devices to cover " << choice.second << " stages" << std::endl;

        std::cout << "Try to Assign Devices: ";
        for (int j = 0; j < assigned_device.size(); j++) {
          std::cout << assigned_device[j] << '\t';
        }
        std::cout << "With Execution Result: " << execution_result->to_string()
                  << std::endl;
        update_dp_slot(i, execution_result, assigned_device);
        std::cout << "DP[" << i << "] is updated with " << std::endl;
        std::cout << "Assign Devices: ";
        for (int j = 0; j < dp_[i].first.size(); j++) {
          std::cout << dp_[i].first[j] << '\t';
        }
        std::cout << dp_[i].second->to_string() << std::endl;
      } // for choice
    }
  }     // for i

  // pprint all initial states when debug
  DEBUG_STMT(print());

  assert(dp_[curr_stages.size()].second != nullptr &&
         "final state is infeasible");

  return HeteroPipelineTemplate(
      dp_[curr_stages.size()].second->get_stages(), dp_[curr_stages.size()].second->get_t1(),
      dp_[curr_stages.size()].second->get_t2(), dp_[curr_stages.size()].second->get_t3(),
      dp_[curr_stages.size()].second->get_kstar_latency(),
      dp_[curr_stages.size()].second->get_t(), num_mbatches_,
      layer_execution_results[0]->size(), hetero_node_spec_);
}
} // namespace oobleck