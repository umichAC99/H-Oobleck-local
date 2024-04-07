#ifndef _OOBLECK_PIPELINE_RECOVERY_H_
#define _OOBLECK_PIPELINE_RECOVERY_H_

#include "hetero_pipeline_template.h"
#include "pipeline_template.h"
#include <unordered_map>
namespace oobleck {
class BasePipelineRecoverSolver {
protected:
  struct RecoveryFailException : public std::exception {
    const std::string reason;

    RecoveryFailException(const std::string reason) : reason(reason) {}
    const char *what() const noexcept override { return reason.c_str(); }
  };
  const std::vector<float> scaling_factors_;
  const HeteroNodeSpec &hetero_node_spec_;
  const int num_mbatches_;
  CacheMap dc_cache_;

  std::shared_ptr<oobleck::DCExecutionResult>
  merge_stages(const std::vector<std::shared_ptr<StageExecutionResult>> &stages,
               const int start_stage_idx, const int end_stage_idx,
               const int num_devices, const int node_type_idx,
               const std::shared_ptr<LayerExecutionResults> profile);

  // update dc_cache for homogenous pipeline template
  void update_homo_dc_cache(
      const std::vector<std::shared_ptr<StageExecutionResult>> &stages);

  // update dc_cache after assigning a stage
  void update_dc_cache(
      int idx, const std::vector<std::shared_ptr<StageExecutionResult>> &stages,
      HeteroNodeSpec &left, HeteroNodeSpec &right);

  // Try to assign stages[idx] with a specific node type with assigned_device
  // number of device Return you the Execution Result with such assignment
  std::shared_ptr<oobleck::DCExecutionResult>
  try_assign(int idx, int node_type, int assigned_device,
             std::shared_ptr<LayerExecutionResults> profile,
             std::vector<std::shared_ptr<StageExecutionResult>> &stages,
             const HeteroNodeSpec &left, const HeteroNodeSpec &right);

public:
  BasePipelineRecoverSolver(const std::vector<float> &scaling_factors,
                            const HeteroNodeSpec &hetero_node_spec,
                            const int num_mbatches)
      : scaling_factors_(scaling_factors), hetero_node_spec_(hetero_node_spec),
        num_mbatches_(num_mbatches) {}

  virtual ~BasePipelineRecoverSolver() = default;

  virtual HeteroPipelineTemplate
  solve(const std::vector<PipelineTemplate> &pipeline_templates,
        const std::vector<std::shared_ptr<LayerExecutionResults>>
            &layer_execution_results) = 0;
};

class GreedyPipelineRecoverSolver : public BasePipelineRecoverSolver {
public:
  GreedyPipelineRecoverSolver(const std::vector<float> &scaling_factors,
                              const HeteroNodeSpec &hetero_node_spec,
                              const int num_mbatches)
      : BasePipelineRecoverSolver(scaling_factors, hetero_node_spec,
                                  num_mbatches) {}

  std::shared_ptr<HeteroPipelineTemplate>
  solve_one(const PipelineTemplate &pipeline_templates,
            const std::vector<std::shared_ptr<LayerExecutionResults>>
                &layer_execution_results);

  HeteroPipelineTemplate
  solve(const std::vector<PipelineTemplate> &pipeline_templates,
        const std::vector<std::shared_ptr<LayerExecutionResults>>
            &layer_execution_results) override;
};

class ButtomUpDPPipelineRecoverSolver : public BasePipelineRecoverSolver {
public:
  ButtomUpDPPipelineRecoverSolver(const std::vector<float> &scaling_factors,
                                  const HeteroNodeSpec &hetero_node_spec,
                                  const int num_mbatches)
      : BasePipelineRecoverSolver(scaling_factors, hetero_node_spec,
                                  num_mbatches) {
    dp_choices_.resize(hetero_node_spec.node_specs.size());
    avail_devices_.resize(hetero_node_spec.node_specs.size());
  }

  typedef std::pair<int /*num device*/, int /*num stages that can be covered*/>
      Choice; // use $first number of devices for covering $second stages
  typedef std::vector<std::vector<Choice>>
      DPChoices; // DPChoices[i] is the all possible choices for node type i
  typedef std::vector<int> DeviceResource; // AvailDevices[i] is the number of
                                           // available devices for node type i
  typedef std::pair<DeviceResource, std::shared_ptr<oobleck::DCExecutionResult>>
      DPState; // DPState.first is the number of assigned devices for each node
               // type DPState.second is the execution result for the current
               // state

  DPChoices dp_choices_;
  DeviceResource avail_devices_;
  std::vector<DPState> dp_;
  const PipelineTemplate *longest_pipeline_;

  void update_dp_slot(int idx, std::shared_ptr<DCExecutionResult> new_result,
                      const DeviceResource &devices) {
    assert(idx < dp_.size() && "idx is out of range");
    if (dp_[idx].second == nullptr) {
      dp_[idx] = DPState(devices, new_result);
    } else {
      if (dp_[idx].second->get_t() > new_result->get_t()) {
        dp_[idx] = DPState(devices, new_result);
      }
    }
  }

  bool over_device_limit(const DeviceResource &devices, int node_type_idx, int new_device) {
    return devices[node_type_idx] + new_device > avail_devices_[node_type_idx];
  }

  // pretty print all resources, choices and dp states
  void print();

  // setup avail_devices_ and dp_choices_ based on input data
  void preprocess(const std::vector<std::shared_ptr<LayerExecutionResults>>
                      &layer_execution_results);

  HeteroPipelineTemplate
  solve(const std::vector<PipelineTemplate> &pipeline_templates,
        const std::vector<std::shared_ptr<LayerExecutionResults>>
            &layer_execution_results) override;
};
} // namespace oobleck

#endif // _OOBLECK_PIPELINE_RECOVERY_H_