# Robot models

Pinned Unitree G1 descriptions and the manifest that binds them to the
calibration release.

## Contents

| Path | Responsibility |
|---|---|
| [`MODEL_MANIFEST.json`](MODEL_MANIFEST.json) | Profile identifiers, file hashes, model paths, and scene paths |
| [`g1/`](g1/) | 29-DoF URDF, MJCF, STL meshes, contact-frame definition, and licenses |

Every shipped mesh was matched against a pinned Unitree repository. Source
commits, adaptations, masses, and preserved BSD 3-Clause license texts are
documented in [`g1/NOTICE.md`](g1/NOTICE.md).

Return to the [G1 implementation](../README.md).
