#include "pipeline_recovery.h"
namespace oobleck {
HeteroPipelineTemplate GreedyPipelineRecoverSolver::solve() const{


    return HeteroPipelineTemplate(
        pipeline_template_.get_stages(),
        pipeline_template_.get_t1(),
        pipeline_template_.get_t2(),
        pipeline_template_.get_t3(),
        pipeline_template_.get_kstar_latency(),
        pipeline_template_.get_iteration_time(),
        pipeline_template_.get_num_mbatches(),
        pipeline_template_.get_num_layers(),
        hetero_node_spec_
    );
}
}