import pytest
import sys

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplateGenerator,
    GreedyPipelineRecoverSolver,
)
from tests.conftest import OobleckSingleProcessTestCase


class TestOobleckPipelineTemplate(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def profile(self) -> LayerExecutionResults:
        return self.factory.get_dummy_profile()   
    
    def test_node_folding(self):
        generator = PipelineTemplateGenerator()
        node_spec = self.factory.get_dummy_hetero_node_spec()
        profiles = self.factory.get_dummy_profile_by_scaling(node_spec)
        # print(node_spec)
        # print(profiles)
        # print("LOG: running ground truth")
        # pipeline_template = generator.create_hetero_pipeline_template(
        #     profiles,
        #     node_spec,
        #     32,
        # )
        # print(pipeline_template)
        print("LOG: running node folding")
        (num_nodes, num_gpus_per_node, scaling_factors) = self.factory.dummy_node_folding(profiles, node_spec)
        print("num_nodes: ", num_nodes)
        print("num_gpus_per_node: ", num_gpus_per_node)
        print("scaling_factors: ", scaling_factors)
        print("num layers: ", len(profiles[0].get()))
        # flush print buffer
        sys.stdout.flush()
        pipeline_template_origin = generator.create_pipeline_templates(
            profiles[0],
            (num_nodes, num_nodes),  # num nodes range
            num_gpus_per_node,
            32,
        )[0]
        print(pipeline_template_origin)
        solver = GreedyPipelineRecoverSolver(pipeline_template_origin, scaling_factors, node_spec, 32)
        plan = solver.solve(profiles)
        print("approximated plan ", plan)
        
        pipeline_template = generator.create_hetero_pipeline_template(
            profiles,
            node_spec,
            32,
        )
        print("real plan ", pipeline_template)
        # plan = recovery(pipeline_template_origin, scaling_factors, node_spec)
        # compare(plan, pipeline_template)
        
    def test_hetero_node_spec(self, random: bool=False, num_nodes: int=5): # num_nodes will not work if not random
        node_spec = self.factory.get_hetero_node_spec(is_random=random, num_nodes=num_nodes)

        print(node_spec)
        assert node_spec.size() > 0
        assert len(node_spec.get()) > 0
        if (random):
            assert node_spec.size() == num_nodes
        else:
            assert node_spec.size() == 5
        num_gen_nodes: int = 0
        for node_config in node_spec.get():
            num_gen_nodes += node_config._num_nodes
            assert node_config._num_gpus > 0
            assert node_config._num_nodes > 0
        assert num_gen_nodes == num_nodes, "#generated nodes should match args"
        
    
    def test_hetero_node_spec_random(self, num_nodes: int=5):
        return self.test_hetero_node_spec(random=True, num_nodes=num_nodes)
    
    @pytest.mark.skip(reason="Skipped")
    def test_create_hetero_pipeline_templates(self):
        generator = PipelineTemplateGenerator()
        hetero_profiles = self.factory.get_dummy_hetero_profile()
        node_spec = self.factory.get_dummy_hetero_node_spec()
        pipeline_template = generator.create_hetero_pipeline_template(
            hetero_profiles,
            node_spec,
        )
        print(pipeline_template)
    
    @pytest.mark.skip(reason="Skipped")
    def test_create_pipeline_templates_onegpu(self, profile: LayerExecutionResults):
        generator = PipelineTemplateGenerator()
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (8, 8),  # num nodes range
            4,
            60
        )
        assert len(pipeline_templates) == 1
        assert pipeline_templates[0]._num_nodes == 8
        assert pipeline_templates[0]._num_gpus_per_node == 4
        # assert len(pipeline_templates[0].get_stages()) == 1
        assert pipeline_templates[0]._iteration_time > 0
        print(pipeline_templates[0])

    @pytest.mark.skip(reason="Skipped")
    def test_create_pipeline_templates_maxnode(self, profile: LayerExecutionResults):
        generator = PipelineTemplateGenerator()
        num_nodes = profile.size  # num_nodes == number of layers
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (num_nodes, num_nodes),  # num nodes range
            1,
        )
        assert len(pipeline_templates) == 1
        assert pipeline_templates[0]._num_nodes == num_nodes
        assert pipeline_templates[0]._num_gpus_per_node == 1
        assert len(pipeline_templates[0].get_stages()) == num_nodes
        assert pipeline_templates[0]._iteration_time > 0

    @pytest.mark.skip(reason="Skipped")
    def test_create_pipeline_templates_too_many_nodes(
        self, profile: LayerExecutionResults
    ):
        generator = PipelineTemplateGenerator()
        num_nodes = profile.size + 1
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (num_nodes, num_nodes),  # num nodes range
            1,
        )
        assert len(pipeline_templates) == 0

    @pytest.mark.skip(reason="Skipped")
    def test_create_pipeline_templates_node_range(self, profile: LayerExecutionResults):
        generator = PipelineTemplateGenerator()
        max_num_nodes = profile.size
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (2, 8),  # num nodes range
            1,
        )
        assert 0 < len(pipeline_templates) <= max_num_nodes
        assert 0 < pipeline_templates[0]._num_nodes <= max_num_nodes
        for pipeline_template in pipeline_templates:
            assert pipeline_templates[0]._num_gpus_per_node == 1
            assert 2 <= len(pipeline_template.get_stages()) <= 8
            assert pipeline_template._iteration_time > 0

    @pytest.mark.skip(reason="Skipped")
    def test_create_pipeline_templates_multiple_gpus_in_node(
        self, profile: LayerExecutionResults
    ):
        generator = PipelineTemplateGenerator()
        pipeline_templates = generator.create_pipeline_templates(profile, (1, 1), 4)
        assert len(pipeline_templates) >= 1
        sum(
            template._num_gpus_per_node * template._num_nodes
            for template in pipeline_templates
        ) == 4
    @pytest.mark.skip(reason="Skipped")
    def test_create_pipeline_templates_multiple_gpus_in_node_range(
        self, profile: LayerExecutionResults
    ):
        generator = PipelineTemplateGenerator()
        pipeline_templates = generator.create_pipeline_templates(profile, (1, 6), 4)
        assert len(pipeline_templates) >= 1
        for index, template in enumerate(pipeline_templates):
            num_nodes = index + 1
            assert template._num_gpus_per_node == 4
            assert num_nodes == template._num_nodes

    @pytest.mark.skip(reason="Not implemented yet")
    def test_create_pipeline_templates_fsdp(self, profile: LayerExecutionResults):
        pass
