from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    get_profile_results,
)
import numpy as np

def get_1000_layers_results() -> LayerExecutionResults:
    profiles: LayerExecutionResults = get_profile_results(
            model_name='gpt2-xl',
            model_tag='EQaPKriX',
            microbatch_size=1,
            node_type="",
        )
    profilesList = profiles.get()
    print(len(profilesList))
    usable_idx = []
    for i, v in enumerate(profilesList):
        if (v._forward > 20):
            usable_idx.append(i)
    
    print("Attention layers idx:")
    print(usable_idx)
    print(f"Length: {len(usable_idx)}")
    print(f"Length total: {len(profilesList)}")
    num_gen = 1000 - len(profilesList) - 1
    print(f"Need generate {num_gen}")
    

    begin: int = usable_idx[0]
    end: int = usable_idx[-1]

    fwd = []
    rdu_i = []
    rdu_c = []
    for idx in usable_idx:
        item = profilesList[idx]
        fwd.append(item._forward)
        rdu_i.append(item._allreduce_in_node[1])
        rdu_c.append(item._allreduce_across_nodes[1])

    print(fwd)
    print(rdu_i)
    print(rdu_c)

    mu_fwd = np.mean(fwd)
    var_fwd = np.var(fwd)

    mu_rdu_i = np.mean(rdu_i)
    var_rdu_i = np.var(rdu_i)

    mu_rdu_c = np.mean(rdu_c)
    var_rdu_c = np.var(rdu_c)

    print(mu_fwd, mu_rdu_i, mu_rdu_c)
    print(var_fwd, var_rdu_i, var_rdu_c)

    new_fwd = np.random.normal(mu_fwd, np.sqrt(var_fwd), num_gen).tolist()
    new_rdu_i = np.random.normal(mu_rdu_i, np.sqrt(var_rdu_i), num_gen).tolist()
    new_rdu_c = np.random.normal(mu_rdu_c, np.sqrt(var_rdu_c), num_gen).tolist()

    fwd += new_fwd
    rdu_i += new_rdu_i
    rdu_c += new_rdu_c

    print(f"Length after generation: {len(fwd)}")
    assert len(fwd) == len(rdu_i) == len(rdu_c)

    layer_idx = profilesList[begin]._index

    print(begin, end)
    print(layer_idx)
    results: list[LayerExecutionResult] = []
    for i in range(0, begin):
        results.append(profilesList[i])
    for item in zip(fwd, rdu_i, rdu_c):
        results.append(LayerExecutionResult(
            layer_index=layer_idx,
            forward=item[0],
            backward=item[0] * 3,
            allreduce_in_node={1 : item[1]},
            allreduce_across_nodes={1: item[2]},
            mem_required=[127208448, 6553600], # hard coded
        ))
        layer_idx += 1
    for i in range(end, len(profilesList)):
        item = profilesList[i]
        results.append(LayerExecutionResult(
            layer_index=layer_idx,
            forward=item._forward,
            backward=item._backward,
            allreduce_in_node=item._allreduce_in_node,
            allreduce_across_nodes=item._allreduce_across_nodes,
            mem_required=item._mem_required
        ))
        layer_idx += 1
    
    result_obj = LayerExecutionResults(results)
    return result_obj

if __name__ == '__main__':
    get_1000_layers_results()