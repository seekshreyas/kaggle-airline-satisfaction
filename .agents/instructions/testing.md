# Testing conventions

- **Test inputs are declarative YAML, not hard-coded Python.** Put cases in `tests/cases/<module>.yaml`, grouped by the function under test. Each case has an `id` (shown in pytest output), its inputs, and either `expected` or `raises: {error, match}`.
- Test files are thin, generic runners: load cases with `tests.case_loader.load_cases(file, section)` and check errors with `expectation(case)`. Adding a case should mean editing YAML only; Python changes only when a new function is tested.
- Use synthetic data only (shapes and vocabularies can come from YAML). Column *names* from the competition are fine; rows are not.
- Test the properties that matter, not just happy paths:
  - leakage (anything learned is fit on the training fold only)
  - determinism (same salt ⇒ same splits, independent of row order)
  - exact stratification and fold coverage (every row validated once)
  - train/test column consistency, and unseen categories don't crash
  - output validity (e.g. submission ids, ranges, NaNs)
- Pin values that must never drift silently (e.g. hash outputs) so an algorithm change is caught.
- When a test fails, decide whether the code or the expectation is wrong, and say which. Sanity-check new tests by breaking an expectation once and confirming they fail.
- Run everything with `pixi run test`.
