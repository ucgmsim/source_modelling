import numpy as np

from source_modelling import rupture_propagation, sources


def test_simple_line_case():
    """Tests a simple rupture propagation scenario.

    Scenario roughly looks like:

            +-+ c
            | |
            +-+
        b +-+  +--------+ d
          +-+  +--------+
      +---+
      |   | a
      |   |
      |   |
      +---+
    """
    plane_a = sources.Plane.from_corners(
        np.array(
            [
                [-45.0304811338428, 169.98462937445362, 0],
                [-45.390362978215435, 169.98462937445362, 0],
                [-45.390362978215435, 170.27941127890773, 0],
                [-45.0304811338428, 170.27941127890773, 0],
            ]
        )
    )
    plane_b = sources.Plane.from_corners(
        np.array(
            [
                [-44.86621465166093, 170.2402902532466, 0],
                [-44.997796634347004, 170.2402902532466, 0],
                [-44.997796634347004, 170.42184906066956, 0],
                [-44.86621465166093, 170.42184906066956, 0],
            ]
        )
    )
    plane_c = sources.Plane.from_corners(
        np.array(
            [
                [-44.81884296158487, 170.39890952912066, 0],
                [-44.81884296158487, 170.52620234810502, 0],
                [-44.552027768554936, 170.52620234810502, 0],
                [-44.552027768554936, 170.39890952912066, 0],
            ]
        )
    )
    plane_d = sources.Plane.from_corners(
        np.array(
            [
                [-44.77558750315725, 170.59964520496885, 0],
                [-44.87057622488399, 170.59964520496885, 0],
                [-44.87057622488399, 170.9225453010559, 0],
                [-44.77558750315725, 170.9225453010559, 0],
            ]
        )
    )
    source_map = {"a": plane_a, "b": plane_b, "c": plane_c, "d": plane_d}
    rupture_causality_tree = (
        rupture_propagation.estimate_most_likely_rupture_propagation(source_map, "a")
    )
    assert rupture_causality_tree == {"a": None, "b": "a", "c": "b", "d": "b"}
    # We don't really care what the output is here because it is just a thin wrapper over the closest points between sources code, which we have already thoroughly tested.
    # We just check here it doesn't crash and outputs vaguely sensible values.
    jump_points = rupture_propagation.jump_points_from_rupture_tree(
        source_map, rupture_causality_tree
    )
    assert len(jump_points) == 3
    for fault_name, jump_pair in jump_points.items():
        # check that the rupture tree does not make impossible jumps (i.e. distance pruning works).
        assert (
            rupture_propagation.distance_between(
                source_map[rupture_causality_tree[fault_name]],
                source_map[fault_name],
                jump_pair.from_point,
                jump_pair.to_point,
            )
            < 15000
        )
