# nalgebra-spacetime

Spacetime Extension for [nalgebra]

[nalgebra]: https://nalgebra.org

[![Build Status][]](https://travis-ci.org/qu1x/nalgebra-spacetime)
[![Downloads][]](https://crates.io/crates/nalgebra-spacetime)
[![Rust][]](https://www.rust-lang.org)
[![Version][]](https://crates.io/crates/nalgebra-spacetime)
[![Documentation][]](https://doc.qu1x.dev/nalgebra-spacetime)
[![License][]](https://opensource.org/licenses/BSD-3-Clause)

[Build Status]: https://travis-ci.org/qu1x/nalgebra-spacetime.svg
[Downloads]: https://img.shields.io/crates/d/nalgebra-spacetime.svg
[Rust]: https://img.shields.io/badge/rust-stable-brightgreen.svg
[Version]: https://img.shields.io/crates/v/nalgebra-spacetime.svg
[Documentation]: https://docs.rs/nalgebra-spacetime/badge.svg
[License]: https://img.shields.io/crates/l/nalgebra-spacetime.svg

API Documentation with KaTeX: [stable] | [master]

[stable]: https://docs.rs/nalgebra-spacetime
[master]: https://doc.qu1x.dev/nalgebra-spacetime

# Present Features

  * Minkowski space as special case of `LorentzianN` space.
  * Raising/Lowering tensor indices: `dual()`/`r_dual()`/`c_dual()`.
  * Metric contraction of degree-1/degree-2 tensors: `contr()`/`scalar()`.
  * Spacetime `interval()` with `LightCone` depiction.
  * Inertial `FrameN` of reference holding boost parameters.
  * Lorentz boost as `new_boost()` matrix.
  * Direct Lorentz `boost()` to `compose()` velocities.
  * Wigner `rotation()` and `axis_angle()` between to-be-composed boosts.

# Future Features

  * `Event4`/`Velocity4`/`Momentum4`/`...` equivalents of `Point4`/`...`.
  * Categorize `Rotation4`/`PureBoost4`/`...` as `Boost4`/`...`.
  * Wigner `rotation()` and `axis_angle()` of an already-composed `Boost4`.
  * Distinguish pre/post-rotation and active/passive `Boost4` compositions.
  * Spacetime algebra (STA) as special case of `CliffordN` space.

# Pseudo-local Documentation Builds

```sh
# Build and open documentation inclusive dependencies.
cargo doc --open
# Update any source file, then regenerate this crate's documentation with KaTeX.
cargo tex
# Refresh opened documentation.
```

With `cargo tex` defined in [.cargo/config](.cargo/config). Note that navigating
the documentation requires web access as KaTeX is embedded via remote CDN.

## License

[BSD-3-Clause](LICENSE.md)

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the works by you shall be licensed as above, without any
additional terms or conditions.
