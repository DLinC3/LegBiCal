# Configuration

Versioned configuration consumed by the G1 lower solver and visualization
paths.

## Contents

| Path | Responsibility |
|---|---|
| [`lower/`](lower/) | H=501 PRIME FDDP/contact-Newton template |
| [`replay/`](replay/) | G1 MuJoCo replay scene aligned with PRIME's checker, skybox, and light |
| [`visualization/`](visualization/) | Meshcat/MuJoCo style v2, force geometry, colors, and floor palette |

Paths embedded in generated lower configurations are resolved through the
repository-root policy in `g1cal.paths`.

Return to the [G1 implementation](../README.md).
