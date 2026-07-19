# Motion clip provenance

The two 10-second, 50 Hz clips are MuJoCo rollouts tracking the Unitree G1
retargets named `run1_subject2` and `run2_subject1`. Both clips are used by the
released covariance-calibration objective; neither is presented as evidence
beyond these two trajectories.

Source chain:

1. Ubisoft La Forge Animation Dataset (LAFAN1), `run1_subject2` and
   `run2_subject1`, published with the *Robust Motion In-Betweening* dataset.
2. Unitree G1 retargets distributed by
   `lvhaidong/LAFAN1_Retargeting_Dataset` on Hugging Face.
3. Tracking-policy MuJoCo rollouts captured at 50 Hz, followed by the released
   deterministic sensor-noise realization and 501-state slices.

The upstream LAFAN1 repository states CC BY-NC-ND 4.0. Its no-derivatives
condition creates a redistribution restriction for retargeted trajectories and
their rollout-derived state slices, regardless of the more permissive label on
the intermediate retarget repository. The CSV and NPZ files in this directory
are therefore local release candidates only. Do not publish them without
written permission from the original motion rightsholder or a documented legal
determination that permits this distribution.

Each `injection.json` records the frozen noise realization, source row range,
state/control alignment, and baseline injection covariance. The covariance in
that file describes data generation; the estimator covariance selected by the
released calibration is stored separately in `data/calibrated/`.

Requested citations:

- Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal,
  “Robust Motion In-Betweening,” ACM Transactions on Graphics 39(4), 2020.
- Ubisoft La Forge Animation Dataset:
  <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>
- G1 retarget collection:
  <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>
