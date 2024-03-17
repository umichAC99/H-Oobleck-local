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

#define DEBUG

/**
 * Extension of Section 4.1.2. GPU-Stage Mapping using divide and conquer
 * algorithm for heterogenous system. The divide and conquer is accelerated
 * using multithreading and memoization.
 */

namespace oobleck {

std::vector<SingleNodeSpec> node_specs;

void generateSubsetsUtil(const HeteroNodeSpec& originalSpec, std::vector<HeteroNodeSpec>& allSubsets, HeteroNodeSpec currentSubset, int start) {
    for (int i = start; i < originalSpec.node_specs.size(); ++i) {
        for (int j = 1; j <= originalSpec.node_specs[i].num_nodes; ++j) {
            HeteroNodeSpec newSubset = currentSubset;
            newSubset.node_specs[i].num_nodes = j;
            newSubset.update_fields();
            allSubsets.push_back(newSubset);
            generateSubsetsUtil(originalSpec, allSubsets, newSubset, i + 1);
        }
    }
}

std::vector<HeteroNodeSpec> generateSubsets(const HeteroNodeSpec& heteroSpec) {
    std::vector<HeteroNodeSpec> allSubsets;
    HeteroNodeSpec currentSubset = heteroSpec;
    for (int i = 0; i < heteroSpec.node_specs.size(); ++i) {
        currentSubset.node_specs[i].num_nodes = 0;
    }
    generateSubsetsUtil(heteroSpec, allSubsets, currentSubset, 0);

    #ifdef DEBUG
    // Print all subsets for demonstration purposes
    for (const auto& subset : allSubsets) {
        std::cout << subset.to_string() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Total number of subsets: " << allSubsets.size() << std::endl;

    // Check there are no duplicates
    for (int i = 0; i < allSubsets.size(); ++i) {
        for (int j = i + 1; j < allSubsets.size(); ++j) {
            assert(allSubsets[i].node_specs != allSubsets[j].node_specs);
        }
    }
    #endif
    return allSubsets;
}

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
  return layer_execution_results;
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

    generateSubsets(node_spec);
    return HeteroPipelineTemplate();
      }

} // namespace oobleck