# nalgebra-spacetime

Spacetime Extension for [nalgebra]

[nalgebra]: https://nalgebra.org

[![Build][]](https://github.com/qu1x/nalgebra-spacetime/actions/workflows/build.yml)
[![Documentation][]](https://docs.rs/nalgebra-spacetime)
[![Downloads][]](https://crates.io/crates/nalgebra-spacetime)
[![Version][]](https://crates.io/crates/nalgebra-spacetime)
[![Rust][]](https://www.rust-lang.org)
[![License][]](https://opensource.org/licenses)

[Build]: https://github.com/qu1x/nalgebra-spacetime/actions/workflows/build.yml/badge.svg
[Documentation]: https://docs.rs/nalgebra-spacetime/badge.svg
[Downloads]: https://img.shields.io/crates/d/nalgebra-spacetime.svg
[Version]: https://img.shields.io/crates/v/nalgebra-spacetime.svg
[Rust]: https://img.shields.io/badge/rust-stable-brightgreen.svg
[License]: https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg

# Present Features

  * Minkowski space as special case of n-dimensional `Lorentzian` space.
  * Raising/Lowering tensor indices: `dual()`/`r_dual()`/`c_dual()`.
  * Metric contraction of degree-1/degree-2 tensors: `contr()`/`scalar()`.
  * Spacetime `interval()` with `LightCone` depiction.
  * Inertial `OFrame` of reference holding boost parameters.
  * Lorentz boost as `new_boost()` matrix.
  * Direct Lorentz `boost()` to `compose()` velocities.
  * Wigner `rotation()` and `axis_angle()` between to-be-composed boosts.

# Future Features

  * `Event4`/`Velocity4`/`Momentum4`/`...` equivalents of `Point4`/`...`.
  * Categorize `Rotation4`/`PureBoost4`/`...` as `Boost4`/`...`.
  * Wigner `rotation()` and `axis_angle()` of an already-composed `Boost4`.
  * Distinguish pre/post-rotation and active/passive `Boost4` compositions.

# Pseudo-local Documentation Builds

```sh
# Build and open documentation inclusive dependencies.
cargo doc --open
# Rebuild this crate's documentation with KaTeX.
cargo tex
# Refresh opened documentation.
```

With `cargo tex` defined in [.cargo/config.toml](.cargo/config.toml). Note that navigating the
documentation requires web access as KaTeX is embedded via remote CDN.

# License

Copyright © 2020,2021,2024 Rouven Spreckels <rs@qu1x.dev>

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSES/Apache-2.0](LICENSES/Apache-2.0) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSES/MIT](LICENSES/MIT) or https://opensource.org/licenses/MIT)

at your option.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
