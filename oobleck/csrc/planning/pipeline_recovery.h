#ifndef _OOBLECK_PIPELINE_RECOVERY_H_
#define _OOBLECK_PIPELINE_RECOVERY_H_

#include "hetero_pipeline_template.h"
#include "pipeline_template.h"
#include <map>
namespace oobleck {
class BasePipelineRecoverSolver {
protected:
  const PipelineTemplate &pipeline_template_;
  const std::vector<float> scaling_factors_;
  const HeteroNodeSpec &hetero_node_spec_;
  const int num_mbatches_;
  CacheMap *dc_cache_ = nullptr;

  void update_dc_cache(
      int idx, const std::vector<std::shared_ptr<StageExecutionResult>> &stages,
      HeteroNodeSpec &left, HeteroNodeSpec &right);

  std::shared_ptr<oobleck::DCExecutionResult>
  try_assign(int idx, int node_type, int assigned_device,
             std::shared_ptr<LayerExecutionResults> profile,
             std::vector<std::shared_ptr<StageExecutionResult>> &stages,
             const HeteroNodeSpec &left, const HeteroNodeSpec &right);

public:
  BasePipelineRecoverSolver(const PipelineTemplate &pipeline_template,
                            const std::vector<float> &scaling_factors,
                            const HeteroNodeSpec &hetero_node_spec,
                            const int num_mbatches)
      : pipeline_template_(pipeline_template),
        scaling_factors_(scaling_factors), hetero_node_spec_(hetero_node_spec),
        num_mbatches_(num_mbatches) {}

  virtual ~BasePipelineRecoverSolver() = default;

  virtual HeteroPipelineTemplate
  solve(const std::vector<std::shared_ptr<LayerExecutionResults>>
            &layer_execution_results) = 0;

  void set_dc_cache(PipelineTemplateGenerator &ptg) {
    dc_cache_ = ptg.get_dc_cache();
    PRINT("Finished Setting DC Cache ");
  }
};

class GreedyPipelineRecoverSolver : public BasePipelineRecoverSolver {
public:
  GreedyPipelineRecoverSolver(const PipelineTemplate &pipeline_template,
                              const std::vector<float> &scaling_factors,
                              const HeteroNodeSpec &hetero_node_spec,
                              const int num_mbatches)
      : BasePipelineRecoverSolver(pipeline_template, scaling_factors,
                                  hetero_node_spec, num_mbatches) {}

  HeteroPipelineTemplate
  solve(const std::vector<std::shared_ptr<LayerExecutionResults>>
            &layer_execution_results) override;
};
} // namespace oobleck

#endif // _OOBLECK_PIPELINE_RECOVERY_H_