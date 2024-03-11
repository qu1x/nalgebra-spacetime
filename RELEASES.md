# Version 0.5.0 (2024-03-11)

  * Replace `N: SimdRealField + Signed + Real` with `T: RealField`.
  * Rename `LorentzianN` to `Lorentzian`.
  * Rename `FrameN` to `OFrame`.
  * Rename `MomentumN` to `OMomentum`.
  * Make `OFrame::rotation()` robust close to identity.

# Version 0.4.0 (2024-02-27)

  * Make Wigner rotation n-dimensional.
  * Negate `FrameN` axis instead of beta.
  * Escape prime symbol.
  * Re-export dependencies.
  * Add 5- and 6-dimensional aliases.

# Version 0.3.0 (2024-02-21)

  * Update dependencies.
  * Change license to `MIT OR Apache-2.0`.

# Version 0.2.4 (2021-03-06)

  * Update `nalgebra`.

# Version 0.2.3 (2021-01-08)

  * Use `intra_rustdoc_links`.
  * Update `nalgebra` and `approx`.

# Version 0.2.2 (2020-08-28)

  * Fix `num-traits` features.
  * Update `nalgebra` and `katex`.

# Version 0.2.1 (2020-06-27)

  * Fix `FrameN::compose()` order.
  * Add `Frame4::rotation()` test.

# Version 0.2.0 (2020-06-27)

  * Overall avoid `unsafe` code without reasoning via benchmarks.
  * Add spacetime `split()/from_split()` and `split_mut()`.
  * Improve `Frame4::axis_angle()` test.
  * Add `boost_mut()` test.
  * Add `Momentum4` (incomplete).
  * Add `temporal()` and `spatial()` methods.
  * Fix signature of norm methods.
  * Clean up trait bounds.

# Version 0.1.2 (2020-06-17)

  * Fix possible unsound iteration.
  * Simplify `interval()`.

# Version 0.1.1 (2020-06-08)

  * Fix `contr()` and `tr_contr()`.
  * Fix license badge link.
  * Add documentation link.
  * Update description.

# Version 0.1.0 (2020-06-08)

  * Experimental release.
