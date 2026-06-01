"""Tests for oracle.cam.boolean_network."""

import numpy as np
import pytest


def test_boolean_simulator_init(tiny_grn, cam_config):
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    sim = BooleanNetworkSimulator(tiny_grn, cam_config)
    assert len(sim.genes) == tiny_grn.number_of_nodes()


def test_gene_index_consistency(tiny_grn, cam_config):
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    sim = BooleanNetworkSimulator(tiny_grn, cam_config)
    for gene in sim.genes:
        assert gene in sim.gene_idx
    assert len(sim.gene_idx) == len(sim.genes)


def test_find_attractors_returns_list(tiny_grn, cam_config):
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    sim = BooleanNetworkSimulator(tiny_grn, cam_config)
    attractors = sim.find_attractors(n_initial_states=20)
    assert isinstance(attractors, list)


def test_attractors_are_binary(tiny_grn, cam_config):
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    sim = BooleanNetworkSimulator(tiny_grn, cam_config)
    attractors = sim.find_attractors(n_initial_states=20)
    for attr in attractors:
        arr = np.array(attr)
        assert arr.shape == (len(sim.genes),)
        assert set(arr.flatten().tolist()).issubset({0.0, 1.0, 0, 1})


def test_attractors_are_fixed_points(tiny_grn, cam_config):
    """Each attractor should be a fixed point: running one more step yields same state."""
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    sim = BooleanNetworkSimulator(tiny_grn, cam_config)
    attractors = sim.find_attractors(n_initial_states=30)
    for attr in attractors:
        next_state, _ = sim._run_trajectory(attr.copy(), max_steps=5)
        np.testing.assert_array_equal(np.array(attr), np.array(next_state))


def test_basin_sizes_sum_to_one(tiny_grn, cam_config):
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    sim = BooleanNetworkSimulator(tiny_grn, cam_config)
    attractors = sim.find_attractors(n_initial_states=30)
    if len(attractors) == 0:
        pytest.skip("No attractors found in tiny GRN")
    basins = sim.compute_basin_sizes(attractors, n_samples=100)
    total = sum(basins.values())
    assert abs(total - 1.0) < 0.01, f"Basin sizes sum to {total}, expected ~1.0"
