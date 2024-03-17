#include "pipeline_template.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cppcoro/when_all.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <ranges>
#include <string>

#ifdef PYBIND11_MODULE
#include <pybind11/pybind11.h>
#endif

/**
 * Extension of Section 4.1.2. GPU-Stage Mapping using divide and conquer
 * algorithm for heterogenous system. The divide and conquer is accelerated
 * using multithreading and memoization.
 */

namespace oobleck {
std::vector<std::shared_ptr<LayerExecutionResults>>
get_hetero_profile_results(const std::vector<std::string> &model_names,
                           const std::vector<std::string> &model_tags,
                           const int microbatch_size,
                           const std::vector<std::string> &node_types) {
  std::vector<std::shared_ptr<LayerExecutionResults>> layer_execution_results;
  for (int i = 0; i < node_types.size(); i++) {
    layer_execution_results.push_back(get_profile_results(
        model_names[i], model_tags[i], microbatch_size, node_types[i]));
  }
}

  HeteroPipelineTemplate
  PipelineTemplateGenerator::create_hetero_pipeline_template(
      std::vector<std::shared_ptr<LayerExecutionResults>>
          layer_execution_results,
      const HeteroNodeSpec& node_spec){
        #ifdef PYBIND11_MODULE
  // Release GIL
  pybind11::gil_scoped_release release;
#endif

    return HeteroPipelineTemplate();
      }

} // namespace oobleck