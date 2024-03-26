#ifndef _OOBLECK_PIPELINE_RECOVERY_H_
#define _OOBLECK_PIPELINE_RECOVERY_H_

#include "pipeline_template.h"
#include "hetero_pipeline_template.h"
namespace oobleck {
class BasePipelineRecoverSolver{
    protected:
        const PipelineTemplate& pipeline_template_;
        const std::vector<float>& scaling_factors_;
        const HeteroNodeSpec& hetero_node_spec_;
    public:
        BasePipelineRecoverSolver(const PipelineTemplate& pipeline_template, const std::vector<float>& scaling_factors, const HeteroNodeSpec& hetero_node_spec)
            : pipeline_template_(pipeline_template), scaling_factors_(scaling_factors),hetero_node_spec_(hetero_node_spec) {}

        virtual ~BasePipelineRecoverSolver() = default;

        virtual HeteroPipelineTemplate solve() const = 0;
};

class GreedyPipelineRecoverSolver : public BasePipelineRecoverSolver {
    public:
        GreedyPipelineRecoverSolver(const PipelineTemplate& pipeline_template, const std::vector<float>& scaling_factors, const HeteroNodeSpec& hetero_node_spec)
            : BasePipelineRecoverSolver(pipeline_template, scaling_factors, hetero_node_spec) {}

        HeteroPipelineTemplate solve() const override;
};
}



#endif // _OOBLECK_PIPELINE_RECOVERY_H_