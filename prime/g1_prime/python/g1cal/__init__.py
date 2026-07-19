"""Bilevel covariance calibration for a G1 humanoid state estimator.

Lower level: PRIME-based fixed-inertia motion estimator (Crocoddyl FDDP with
a smoothed second-order-cone contact Newton solver). Upper level: SE(3)-log
trajectory loss over a 17-block covariance parameterization, calibrated with
SQP--BFGS and Frank--Wolfe--SDP over one shared variance-coordinate oracle.
"""
