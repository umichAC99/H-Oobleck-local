#include "pipeline_recovery.h"
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
   sorted based on scaling_factors_ in descending order (first one is the
   strongest node)

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

std::shared_ptr<oobleck::DCExecutionResult>
BasePipelineRecoverSolver::try_assign(
    int idx, int assigned_device,
    const std::shared_ptr<LayerExecutionResults> &profile) const {
  return nullptr;
}

HeteroPipelineTemplate GreedyPipelineRecoverSolver::solve() const {

  assert(dc_cache_ != nullptr && "DC Cache is not set");

  return HeteroPipelineTemplate(
      pipeline_template_.get_stages(), pipeline_template_.get_t1(),
      pipeline_template_.get_t2(), pipeline_template_.get_t3(),
      pipeline_template_.get_kstar_latency(),
      pipeline_template_.get_iteration_time(),
      pipeline_template_.get_num_mbatches(),
      pipeline_template_.get_num_layers(), hetero_node_spec_);
}
} // namespace oobleck