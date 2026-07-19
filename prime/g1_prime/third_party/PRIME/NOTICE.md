# Notice

This repository is a standalone research codebase for PRIME, built on
Crocoddyl's optimal-control and FDDP infrastructure.

Crocoddyl is an optimal control library originally developed by the loco-3d
project and contributors. The upstream project is available at:

https://github.com/loco-3d/crocoddyl

The original Crocoddyl copyright notices and BSD-3-Clause license are preserved
in this repository. Unless otherwise stated in individual files, modifications
and additional source files in this repository are distributed under the same
BSD-3-Clause license.

PRIME-specific additions and modifications are:

Copyright (C) 2026 Jiarong Kang, Legged AI Lab,
University of Wisconsin-Madison

This repository adds a contact-identification extension focused on
differentiable contact optimization for estimation and inertial-parameter
identification. Major added or reorganized components include:

- XML-driven contact-identification experiment executable
- Anitescu-style differentiable contact dynamics extensions
- parameter-augmented multibody state and actuation models
- generic contact-ID problem and configuration loaders
- experiment-facing XML configs and preprocessing utilities

This repository is not an official Crocoddyl release and is not maintained by
the upstream Crocoddyl maintainers unless explicitly stated elsewhere.
