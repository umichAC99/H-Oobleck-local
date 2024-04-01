#include "pipeline_recovery.h"
#include <cmath>
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
        &layer_execution_results;
        const CacheMap* dc_cache_;

    @return:
        HeteroPipelineTemplate

    @assumption:
        scaling_factors_, hetero_node_spec_ and layer_execution_results are
   sorted based on scaling_factors_ in ascending order (first one is the
   weakest node)

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

static DCExecutionResult::key get_dc_key(int num_stages, int start_layer_idx,
                                         int end_layer_idx,
                                         const HeteroNodeSpec &spec) {
  std::string device_key;
  device_key = spec.get_cache_key_recovery();
  return std::make_tuple(num_stages, start_layer_idx, end_layer_idx,
                         device_key);
}

// try to assign node idx i with assigned_device to stage, update left and right
// spec
static void update_node_spec(std::shared_ptr<StageExecutionResult> stage,
                             int node_idx, int assigned_device,
                             HeteroNodeSpec &left_spec,
                             HeteroNodeSpec &right_spec) {
  assert(stage->node_type_idx_ == 0 &&
         "Trying to assign a stage that has been assigned");
}

static void empty_node_spec(HeteroNodeSpec &spec) {
  for (int i = 0; i < spec.node_specs.size(); i++) {
    spec.node_specs[i].num_nodes = 0;
    spec.node_specs[i].num_total_gpus = 0;
  }
  spec.num_total_gpus = 0;
  spec.num_total_nodes = 0;
}

static void replace_device(HeteroNodeSpec &spec, int src, int dest,
                           int src_device, int dest_device) {
  spec.node_specs[src].num_total_gpus -= src_device;
  spec.num_total_gpus -= src_device;
  assert(spec.node_specs[src].num_total_gpus >= 0 &&
         "(1)Source device is not enough");
  assert(spec.num_total_gpus >= 0 && "(2)Source device is not enough");
  if (spec.node_specs[src].num_total_gpus > 0) {
    spec.node_specs[src].num_nodes =
        ceil(float(spec.node_specs[src].num_total_gpus) /
             spec.node_specs[src].num_gpus);
  } else {
    spec.node_specs[src].num_nodes = 0;
  }

  spec.node_specs[dest].num_total_gpus += dest_device;
  spec.num_total_gpus += dest_device;
  assert(spec.node_specs[dest].num_total_gpus >= 0 &&
         "Destination device is not enough");
  if (spec.node_specs[dest].num_total_gpus > 0) {
    spec.node_specs[dest].num_nodes =
        ceil(float(spec.node_specs[dest].num_total_gpus) /
             spec.node_specs[dest].num_gpus);
  } else {
    spec.node_specs[dest].num_nodes = 0;
  }
  // PRINT("After Replace Device: " + spec.to_string());
}

static std::string get_cache_key_recovery_merge(
    const HeteroNodeSpec &left, const HeteroNodeSpec &right,
    const std::shared_ptr<StageExecutionResult> current_stage) {
  assert(left.node_specs.size() == right.node_specs.size());
  std::vector<int> num_total_gpus_configs{(int)left.node_specs.size(), 0};
  std::string result = "";
  for (int i = 0; i < left.node_specs.size(); i++) {
    num_total_gpus_configs[i] =
        left.node_specs[i].num_total_gpus + right.node_specs[i].num_total_gpus;
    if (i == current_stage->node_type_idx_) {
      num_total_gpus_configs[i] += current_stage->num_gpus_;
    }

    if (num_total_gpus_configs[i] == 0)
      continue;
    result += DCExecutionResult::get_device_indices_key(
                  num_total_gpus_configs[i], i) +
              "-";
  }
  assert(result.size() > 0 && "Cache key is empty");
  result.pop_back();
  return result;
}

void BasePipelineRecoverSolver::update_homo_dc_cache(
    const std::vector<std::shared_ptr<StageExecutionResult>> &stages) {
  int total_gpu_nums = 0;
  int end_layer_idx;
  for (int i = 0; i < stages.size(); i++) {
    total_gpu_nums = stages[i]->num_gpus_;
    assert(stages[i]->node_type_idx_ == 0 &&
           "It's not a homogenous pipeline template");
    int curr_total_gpu_nums = total_gpu_nums;
    int start_layer_idx = stages[i]->get_start_layer_index();
    auto current_result = std::make_shared<DCExecutionResult>(stages[i]);
    for (int j = i + 1; j < stages.size(); j++) {
      assert(stages[j]->node_type_idx_ == 0 &&
             "It's not a homogenous pipeline template");
      end_layer_idx = stages[j]->get_end_layer_index() + 1;
      curr_total_gpu_nums += stages[j]->num_gpus_;
      current_result = std::make_shared<DCExecutionResult>(
          current_result, std::make_shared<DCExecutionResult>(stages[j]),
          num_mbatches_);
      DCExecutionResult::key key = std::make_tuple(
          j - i + 1, start_layer_idx, end_layer_idx,
          DCExecutionResult::get_device_indices_key(curr_total_gpu_nums, 0));
      // if cannot find the key-val, insert it into cache
      auto it = dc_cache_->find(key);
      if (it == dc_cache_->end()) {
        dc_cache_->insert({key, current_result});
      } else if (it->second->get_t() > current_result->get_t()) {
        it->second = current_result;
      }
    } // for
  }
}

// update dc cache needed for next iteration
void BasePipelineRecoverSolver::update_dc_cache(
    int idx, const std::vector<std::shared_ptr<StageExecutionResult>> &stages,
    HeteroNodeSpec &left_spec, HeteroNodeSpec &right_spec) {
  empty_node_spec(left_spec);
  empty_node_spec(right_spec);

  auto current_stage = stages[idx];
  const auto current_result =
      std::make_shared<DCExecutionResult>(current_stage);
  std::shared_ptr<DCExecutionResult> result_to_cache = nullptr;

  {
    // insert current result to dc_cache
    auto dc_cache_key = std::make_tuple(
        1, current_stage->get_start_layer_index(),
        current_stage->get_end_layer_index() + 1,
        get_cache_key_recovery_merge(left_spec, right_spec, current_stage));
    // PRINT("Current Key: " + DCExecutionResult::key_to_string(dc_cache_key) +
    //       " to result " + current_result->to_string());
    auto it = dc_cache_->find(dc_cache_key);
    assert(it == dc_cache_->end() && "DCExecutionResult already in cache");
    dc_cache_->insert({dc_cache_key, current_result});
  }

  // initialize left and right
  for (int i = 0; i < idx; i++) {
    replace_device(left_spec, 0, stages[i]->node_type_idx_, 0,
                   stages[i]->num_gpus_);
  }
  for (int i = idx + 1; i < stages.size(); i++) {
    replace_device(right_spec, 0, stages[i]->node_type_idx_, 0,
                   stages[i]->num_gpus_);
  }

  auto initialized_right_spec = right_spec;
  // PRINT("Left Spec: " + left_spec.to_string());
  // PRINT("Right Spec: " + right_spec.to_string());

  int i = 0;
  while (i <= idx) {
    std::shared_ptr<oobleck::DCExecutionResult> left_result = nullptr;
    std::shared_ptr<oobleck::DCExecutionResult> right_result = nullptr;

    // get left result if i < idx
    if (i < idx) {
      auto key =
          get_dc_key(idx - i, stages[i]->get_start_layer_index(),
                     stages[idx - 1]->get_end_layer_index() + 1, left_spec);
      // PRINT("Key1: " + DCExecutionResult::key_to_string(key));
      auto it = dc_cache_->find(key);
      if (it != dc_cache_->end()) {
        left_result = it->second;
      } else {
        assert(false && "Left DCExecutionResult not found");
      }
      assert(left_result != nullptr && "Left DCExecutionResult is null");
    }

    // get all possible right results
    for (int j = stages.size() - 1; j > idx; j--) {
      auto key = get_dc_key(j - idx, stages[idx + 1]->get_start_layer_index(),
                            stages[j]->get_end_layer_index() + 1, right_spec);
      // PRINT("Key2: " + DCExecutionResult::key_to_string(key));
      auto it = dc_cache_->find(key);
      if (it != dc_cache_->end()) {
        right_result = it->second;
      } else {
        assert(false && "Right DCExecutionResult not found");
      }

      assert(right_result != nullptr && "Right DCExecutionResult is null");
      int num_stages = (idx - i) + (j - idx) + 1;
      auto dc_cache_key = std::make_tuple(
          num_stages, stages[i]->get_start_layer_index(),
          stages[j]->get_end_layer_index() + 1,
          get_cache_key_recovery_merge(left_spec, right_spec, stages[idx]));
      // PRINT("left spec: " + left_spec.to_string());
      // PRINT("right spec: " + right_spec.to_string());
      // PRINT("Inserting Key: " +
      // DCExecutionResult::key_to_string(dc_cache_key));
      if (left_result != nullptr) {
        result_to_cache = std::make_shared<DCExecutionResult>(
            std::make_shared<DCExecutionResult>(left_result, current_result,
                                                num_mbatches_),
            right_result, num_mbatches_);
      } else {
        result_to_cache = std::make_shared<DCExecutionResult>(
            current_result, right_result, num_mbatches_);
      }

      // insert key pair to dc_cache
      // PRINT(
      //     "Left+Right Key: " + DCExecutionResult::key_to_string(dc_cache_key)
      //     + " to result " + result_to_cache->to_string());
      it = dc_cache_->find(dc_cache_key);
      assert(it == dc_cache_->end() && "DCExecutionResult already in cache");
      dc_cache_->insert({dc_cache_key, result_to_cache});

      // update right spec
      replace_device(right_spec, stages[j]->node_type_idx_, 0,
                     stages[j]->num_gpus_, 0);
    } // for

    // merge left_spec and current result
    if (i < idx) {
      int num_stages = idx - i + 1;
      auto dc_cache_key = std::make_tuple(
          num_stages, stages[i]->get_start_layer_index(),
          stages[idx]->get_end_layer_index() + 1,
          get_cache_key_recovery_merge(left_spec, right_spec, stages[idx]));
      result_to_cache = std::make_shared<DCExecutionResult>(
          left_result, current_result, num_mbatches_);
      // PRINT("Left+null Key: " +
      // DCExecutionResult::key_to_string(dc_cache_key) +
      //       " to result " + result_to_cache->to_string());
      // insert key pair to dc_cache
      auto it = dc_cache_->find(dc_cache_key);
      assert(it == dc_cache_->end() && "DCExecutionResult already in cache");
      dc_cache_->insert({dc_cache_key, result_to_cache});
      replace_device(left_spec, stages[i]->node_type_idx_, 0,
                     stages[i]->num_gpus_, 0);
    }
    // update left spec and right
    right_spec = initialized_right_spec;
    i++;
  } // while
}

std::shared_ptr<oobleck::DCExecutionResult>
BasePipelineRecoverSolver::try_assign(
    int idx, int node_type, int assigned_device,
    std::shared_ptr<LayerExecutionResults> profile,
    std::vector<std::shared_ptr<StageExecutionResult>> &stages,
    const HeteroNodeSpec &left, const HeteroNodeSpec &right) {

  // find DCExecutionResult from 0...idx-1
  std::shared_ptr<oobleck::DCExecutionResult> left_result = nullptr;
  if (idx > 0) {
    auto key =
        get_dc_key(idx, 0, stages[idx - 1]->get_end_layer_index() + 1, left);
    // PRINT("Left Key: " + DCExecutionResult::key_to_string(key));
    auto it = dc_cache_->find(key);
    if (it != dc_cache_->end()) {
      left_result = it->second;
    } else {
      assert(false && "Left DCExecutionResult not found");
    }
  }

  // find DCExecutionResult from idx+1...end
  std::shared_ptr<oobleck::DCExecutionResult> right_result = nullptr;
  if (idx < stages.size() - 1) {
    auto key = get_dc_key(
        stages.size() - idx - 1, stages[idx + 1]->get_start_layer_index(),
        stages[stages.size() - 1]->get_end_layer_index() + 1, right);
    auto it = dc_cache_->find(key);
    // PRINT("Right Key: " + DCExecutionResult::key_to_string(key));
    if (it != dc_cache_->end()) {
      right_result = it->second;
    } else {
      assert(false && "Right DCExecutionResult not found");
    }
  }

  // create new DCExecutionResult based on current assignment
  auto stage = std::make_shared<StageExecutionResult>(
      profile,
      std::make_tuple(stages[idx]->get_start_layer_index(),
                      stages[idx]->get_end_layer_index() + 1),
      assigned_device, node_type);
  auto curr_result = std::make_shared<DCExecutionResult>(stage);
  // PRINT("Curr T " + std::to_string(curr_result->get_t()) + " Curr T1 " +
  //       std::to_string(curr_result->get_t1()) + " Curr T2 " +
  //       std::to_string(curr_result->get_t2()) + " Curr T3 " +
  //       std::to_string(curr_result->get_t3()) + " Curr Kstar " +
  //       std::to_string(curr_result->get_kstar_latency()));

  // merge left and current result
  if (left_result != nullptr) {
    curr_result = std::make_shared<DCExecutionResult>(left_result, curr_result,
                                                      num_mbatches_);
    // PRINT("left T " + std::to_string(left_result->get_t())
    //       + " left T1 " + std::to_string(left_result->get_t1())
    //       + " left T2 " + std::to_string(left_result->get_t2())
    //       + " left T3 " + std::to_string(left_result->get_t3())
    //       + " left Kstar " +
    //       std::to_string(left_result->get_kstar_latency()));
  }

  // merge right and current result
  if (right_result != nullptr) {
    curr_result = std::make_shared<DCExecutionResult>(curr_result, right_result,
                                                      num_mbatches_);
    // PRINT("right T " + std::to_string(right_result->get_t())
    //       + " right T1 " + std::to_string(right_result->get_t1())
    //       + " right T2 " + std::to_string(right_result->get_t2())
    //       + " right T3 " + std::to_string(right_result->get_t3())
    //       + " right Kstar " +
    //       std::to_string(right_result->get_kstar_latency()));
  }
  return curr_result;
}

HeteroPipelineTemplate GreedyPipelineRecoverSolver::solve(
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results) {

  assert(dc_cache_ != nullptr && "DC Cache is not set");
  assert(pipeline_template_.get_num_layers() ==
             layer_execution_results[0]->size() &&
         "Layer Execution Results size is not equal to pipeline template size");
  auto curr_stages = pipeline_template_.get_stages();
  HeteroNodeSpec curr_spec, left_spec, right_spec;

  // update dc cache for current stage
  update_homo_dc_cache(curr_stages);

  // initialize curr_spec to be homogenous cluster
  curr_spec.node_specs = hetero_node_spec_.node_specs;
  for (int i = 0; i < curr_spec.node_specs.size(); i++) {
    if (i == 0) {
      curr_spec.node_specs[i].num_nodes = pipeline_template_.get_num_nodes();
      curr_spec.node_specs[i].num_gpus =
          pipeline_template_.get_num_gpus_per_node();
      curr_spec.node_specs[i].num_total_gpus =
          pipeline_template_.get_num_gpus_per_node() *
          pipeline_template_.get_num_nodes();
    } else {
      curr_spec.node_specs[i].num_nodes = 0;
      curr_spec.node_specs[i].num_gpus =
          pipeline_template_.get_num_gpus_per_node();
      curr_spec.node_specs[i].num_total_gpus = 0;
    }
  }
  curr_spec.update_fields();
  PRINT("Curr Spec: " + curr_spec.to_string());
  PRINT("Scaling Fact: ");
  for (int i = 0; i < scaling_factors_.size(); i++) {
    PRINT(std::to_string(scaling_factors_[i]) + " ");
  }

  // start greedy algorithm
  std::shared_ptr<oobleck::DCExecutionResult> min_cost_dc_result = nullptr;
  std::vector<std::shared_ptr<oobleck::DCExecutionResult>>
      min_cost_dc_result_records;
  for (int i = hetero_node_spec_.node_specs.size() - 1; i > 0; i--) {
    int used_device = 0;
    int total_device = hetero_node_spec_.node_specs[i].num_nodes *
                       hetero_node_spec_.node_specs[i].num_gpus;
    while (used_device < total_device) {
      double min_time = std::numeric_limits<double>::max();
      int min_idx = -1;
      int min_time_assigned_device = -1;
      int assigned_device = -1;
      min_cost_dc_result = nullptr;

      // update left and right ptrs, empty left first
      left_spec = curr_spec;
      right_spec = curr_spec;
      empty_node_spec(left_spec);
      for (int j = 0; j < curr_stages.size(); j++) {

        // assign device to stage based on scaling factor f
        double assigned_device_f =
            curr_stages[j]->num_gpus_ / scaling_factors_[i];
        // PRINT("Assigned Device F: " + std::to_string(assigned_device_f) +
        //       "Scaling Factor: " + std::to_string(scaling_factors_[i]));
        if (assigned_device_f + used_device > total_device)
          assigned_device_f = total_device - used_device;

        assigned_device = ceil(assigned_device_f);
        assert(assigned_device > 0 && "Assigned device is not set");

        // try to assign node idx i with assigned_device to stage, update left
        // and right spec
        int curr_stage_gpu = curr_stages[j]->num_gpus_;
        int curr_stage_node_idx = curr_stages[j]->node_type_idx_;

        // remove current device from right spec
        replace_device(right_spec, curr_stage_node_idx, i, curr_stage_gpu, 0);
        // PRINT("TRY CALL ASSIGN " + std::to_string(j) + " " +
        // std::to_string(i) +
        //       " " + std::to_string(assigned_device) + " " +
        //       curr_spec.to_string() + " " + curr_stages[j]->to_string() + " "
        //       + left_spec.to_string() + " " + right_spec.to_string());

        // only try assign if current stage is assigned to the weakest node
        if (curr_stage_node_idx == 0 && assigned_device_f > 0.8f) {
          auto dc_result =
              try_assign(j, i, assigned_device, layer_execution_results[i],
                         curr_stages, left_spec, right_spec);
          // PRINT("Previous T is " +
          //       std::to_string(pipeline_template_.get_iteration_time()) +
          //       " Previous T1 is " +
          //       std::to_string(pipeline_template_.get_t1()) + " Previous T2
          //       is " + std::to_string(pipeline_template_.get_t2()) + "
          //       Previous T3 is " +
          //       std::to_string(pipeline_template_.get_t3()) + " Previous
          //       Kstar is " +
          //       std::to_string(pipeline_template_.get_kstar_latency()));
          // PRINT("Current T is " + std::to_string(dc_result->get_t()) +
          //       " Current T1 is " + std::to_string(dc_result->get_t1()) +
          //       " Current T2 is " + std::to_string(dc_result->get_t2()) +
          //       " Current T3 is " + std::to_string(dc_result->get_t3()) +
          //       " Current Kstar is " +
          //       std::to_string(dc_result->get_kstar_latency()));
          if (dc_result->get_t() < min_time) {
            min_time = dc_result->get_t();
            min_idx = j;
            min_time_assigned_device = assigned_device;
            min_cost_dc_result = dc_result;
          }
        } // if

        // assign current device to left spec
        replace_device(left_spec, i, curr_stage_node_idx, 0, curr_stage_gpu);
      } // for

      PRINT("[RESULT]: Min Time: " + std::to_string(min_time) + '\n' +
            " Min Idx: " + std::to_string(min_idx) + '\n' +
            " Assigned Device: " + std::to_string(min_time_assigned_device) +
            '\n' + " Used Device: " + std::to_string(used_device) + '\n' +
            " Total Device: " + std::to_string(total_device) + '\n' +
            " Current Spec: " + curr_spec.to_string() + '\n' +
            " Min Cost DC Result: " + min_cost_dc_result->to_string());
      // update current spec and current stages
      replace_device(curr_spec, 0, i, curr_stages[min_idx]->num_gpus_,
                     assigned_device);
      curr_stages = min_cost_dc_result->get_stages();

      // update dc cache with current result
      assert(min_time_assigned_device != -1 && "Assigned device is not set");
      update_dc_cache(min_idx, curr_stages, left_spec, right_spec);
      min_cost_dc_result_records.push_back(min_cost_dc_result);
      used_device += min_time_assigned_device;
    } // while
  }

  return HeteroPipelineTemplate(
      curr_stages, min_cost_dc_result->get_t1(), min_cost_dc_result->get_t2(),
      min_cost_dc_result->get_t3(), min_cost_dc_result->get_kstar_latency(),
      min_cost_dc_result->get_t(), num_mbatches_,
      layer_execution_results[0]->size(), hetero_node_spec_);
}
} // namespace oobleck