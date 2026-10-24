# Releasing GLPlot

A release is a tag on `main`. Publishing to PyPI is automatic once the GitHub Release exists
(`.github/workflows/publish.yml`). A version can be uploaded to PyPI only once, so everything
before the last step is about not having to take it back.

## Before tagging

1. **Version.** `pyproject.toml`, `glplot/__init__.py`, `CITATION.cff` (also `date-released`)
   and `docs/source/conf.py` all carry the same number; `tests/test_version_consistency.py`
   fails if they drift.
2. **CHANGELOG.** Move `[Unreleased]` into a `## [x.y.z] - YYYY-MM-DD` section and update the
   compare links at the bottom.
3. **CI is green on `main`** (Lint, Build, Tests, all nine matrix cells).
4. **Run the OpenGL smoke test on every OS you can reach.** CI runs it on Linux (Mesa software
   rendering), but not on macOS or Windows, and not with a real GPU or window:

   ```bash
   python tools/gl_smoke.py
   ```

   Then open the GUI once (`python -m glplot`) and plot something: the test suite never opens
   a live window.
5. **Build and inspect a fresh `dist/`.**

   ```bash
   rm -rf dist && python -m build && twine check dist/*
   ```

   `dist/` must contain only the new version.

## Tagging

```bash
git tag -a vX.Y.Z -m "GLPlot X.Y.Z"
git push origin vX.Y.Z
```

On GitHub: *Releases -> Draft a new release*, choose the tag, paste the CHANGELOG section as the
notes, and publish. That runs the Publish workflow, which checks that the tag matches the
package version, builds, and uploads to PyPI.

## One-time setup

- **PyPI Trusted Publisher** (https://pypi.org/manage/project/glplot/settings/publishing/):
  owner `AkarisDimitry`, repository `GLPlot`, workflow `publish.yml`, environment `pypi`.
- **GitHub environment** named `pypi` (Settings -> Environments). Add yourself as a required
  reviewer if you want a manual approval before each upload.
- **TestPyPI** (optional dress rehearsal): upload by hand with
  `twine upload --repository testpypi dist/*`, then
  `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ glplot`.
