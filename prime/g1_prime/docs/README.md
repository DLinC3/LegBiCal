# G1 GitHub Pages

Source landing page for **G1 PRIME after covariance calibration**. The two
self-contained Meshcat replays are generated from the shipped calibrated
solutions and are not stored in Git history.

## Contents

| Path | Responsibility |
|---|---|
| [`index.html`](index.html) | Lightweight landing page with one on-demand viewer and direct replay links |

## Build locally

From the G1 implementation directory:

```bash
python scripts/publish_pages.py
```

This creates `out/pages_site/` with the landing page and
`media/run{1,2}_calibrated.html`. Each replay is approximately 54 MB, defaults
to `0.5x`, and accepts `?speed=1`. The landing page keeps at most one replay
loaded at a time; direct links open a replay independently.

## GitHub Pages setting

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Open **Actions**, select **Build and deploy G1 PRIME calibrated pages**, and
   run the workflow manually.

The workflow at
[`../../../.github/workflows/pages.yml`](../../../.github/workflows/pages.yml)
builds and uploads the generated `out/pages_site` tree as one Pages artifact. Its
approximately 104 MiB total is below GitHub Pages' 1 GB published-site limit;
the generated files bypass the 50 MiB regular-Git warning because they are
artifact content, not commits.

See GitHub's official documentation for
[selecting a Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
and [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).

Return to the [G1 implementation](../README.md).
