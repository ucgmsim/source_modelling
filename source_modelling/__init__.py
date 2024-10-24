"""Source Modelling

The source modelling repo is a collection of modules for representing faults, modelling rupture paths, reading source models, and other utilities related to source modelling.

Source Geometry
---------------

The source_modelling package provides many ways to represent fault geometry:

- As point sources, with a small area (`source_modelling.sources.Point`),
- As planes (`source_modelling.sources.Plane`),
- As a connected series of planes (`source_modelling.sources.Fault`).

For each geometry we define common properties (strike, dip, length,
width, etc) and a local coordinate space to convert between points on
the plane and global coordinates.

Rupture Paths
-------------

The `source_modelling.rupture_propagation` module can find likely
rupture paths between faults using a probabilistic approach based on
the closest points between faults. The `workflow` makes use of this to
simulate multi-segment ruptures.


Slip and Moment
---------------

The `source_modelling.slip` and `source_modelling.moment` modules
contain mathematical functions related to `slip` and `moment`
respectively.

Source Formats
--------------
The source modelling package contains modules for parsing common formats for specifying ruptures, including:

- A highly optimised SRF parser (`source_modelling.srf`).
- A FSP parser (`source_modelling.fsp`)
- A GSF parser (`source_modelling.gsf`)"""
