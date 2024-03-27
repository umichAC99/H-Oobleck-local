#ifndef _OOBLECK_PIPELINE_RECOVERY_H_
#define _OOBLECK_PIPELINE_RECOVERY_H_

#include "pipeline_template.h"
#include "hetero_pipeline_template.h"
#include "oobleck_utils.h"
namespace oobleck {
class BasePipelineRecoverSolver{
    protected:
        const PipelineTemplate& pipeline_template_;
        const std::vector<float>& scaling_factors_;
        const HeteroNodeSpec& hetero_node_spec_;
        const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results_;
        const CacheMap* dc_cache_ = nullptr;
    public:
        BasePipelineRecoverSolver(const PipelineTemplate& pipeline_template, const std::vector<float>& scaling_factors, const HeteroNodeSpec& hetero_node_spec, const std::vector<std::shared_ptr<LayerExecutionResults>>
          &layer_execution_results)
            : pipeline_template_(pipeline_template), scaling_factors_(scaling_factors),hetero_node_spec_(hetero_node_spec), layer_execution_results_(layer_execution_results_){}

        virtual ~BasePipelineRecoverSolver() = default;

        virtual HeteroPipelineTemplate solve() const = 0;

        void set_dc_cache(const PipelineTemplateGenerator& ptg) { 
            dc_cache_ = ptg.get_dc_cache(); 
            PRINT("Finished Setting DC Cache ");
        }
};

class GreedyPipelineRecoverSolver : public BasePipelineRecoverSolver {
    public:
        GreedyPipelineRecoverSolver(const PipelineTemplate& pipeline_template, const std::vector<float>& scaling_factors, const HeteroNodeSpec& hetero_node_spec, const std::vector<std::shared_ptr<LayerExecutionResults>>
          &layer_execution_results)
            : BasePipelineRecoverSolver(pipeline_template, scaling_factors, hetero_node_spec, layer_execution_results_) {}

        HeteroPipelineTemplate solve() const override;
};
}



#endif // _OOBLECK_PIPELINE_RECOVERY_H_