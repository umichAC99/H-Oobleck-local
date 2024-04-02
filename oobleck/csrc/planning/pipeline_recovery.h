#ifndef _OOBLECK_PIPELINE_RECOVERY_H_
#define _OOBLECK_PIPELINE_RECOVERY_H_

#include "hetero_pipeline_template.h"
#include "pipeline_template.h"
#include <map>
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
} // namespace oobleck

#endif // _OOBLECK_PIPELINE_RECOVERY_H_