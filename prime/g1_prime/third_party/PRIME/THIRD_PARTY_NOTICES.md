# Third-Party Notices

This repository builds on and depends on several third-party projects. This file
is a high-level attribution guide and does not replace the license files shipped
with those projects.

## Crocoddyl

This repository builds on Crocoddyl and includes modified Crocoddyl source
files.

- Upstream: https://github.com/loco-3d/crocoddyl
- License: BSD-3-Clause

The original Crocoddyl license and copyright notices are preserved in `LICENSE`
and source headers where applicable.

## Pinocchio

Rigid-body dynamics, kinematics, and model parsing are provided through
Pinocchio.

- Upstream: https://github.com/stack-of-tasks/pinocchio

## Eigen

Linear algebra functionality is provided through Eigen.

- Upstream: https://eigen.tuxfamily.org

## Boost

Several C++ utilities and smart pointer interfaces are provided through Boost.

- Upstream: https://www.boost.org

## Ipopt

Ipopt is an optional dependency used by Crocoddyl's solver support when enabled.

- Upstream: https://github.com/coin-or/Ipopt

## example-robot-data

Some benchmark and example programs may use `example-robot-data`.

- Upstream: https://github.com/gepetto/example-robot-data

## User-Provided Robot Assets And Logs

The contact-ID XML interface is designed to reference local user-provided URDF,
SRDF, and CSV data. Private or bulky robot assets and experiment logs should not
be treated as required bundled assets for the public interface unless their
licenses explicitly allow redistribution.
