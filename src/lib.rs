//! Spacetime Extension for [nalgebra]
//!
//! [nalgebra]: https://nalgebra.org
//!
//! # Present Features
//!
//!   * Minkowski space as special case of [`LorentzianN`] space.
//!   * Raising/Lowering tensor indices:
//!     [`LorentzianN::dual`]/[`LorentzianN::r_dual`]/[`LorentzianN::c_dual`].
//!   * Metric contraction of degree-1/degree-2 tensors:
//!     [`LorentzianN::contr`]/[`LorentzianN::scalar`].
//!   * Spacetime [`LorentzianN::interval`] with [`LightCone`] depiction.
//!   * Inertial [`FrameN`] of reference holding boost parameters.
//!   * Lorentz boost as [`LorentzianN::new_boost`] matrix.
//!   * Direct Lorentz [`LorentzianN::boost`] to [`FrameN::compose`] velocities.
//!   * Wigner [`FrameN::rotation`] and [`FrameN::axis_angle`] between to-be-composed boosts.
//!
//! # Future Features
//!
//!   * `Event4`/`Velocity4`/`Momentum4`/`...` equivalents of `Point4`/`...`.
//!   * Categorize `Rotation4`/`PureBoost4`/`...` as `Boost4`/`...`.
//!   * Wigner [`FrameN::rotation`] and [`FrameN::axis_angle`] of an already-composed `Boost4`.
//!   * Distinguish pre/post-rotation and active/passive `Boost4` compositions.

#![forbid(unsafe_code)]
#![forbid(missing_docs)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::type_complexity)]

pub use approx;
pub use nalgebra;
pub use num_traits;

use approx::{abs_diff_eq, AbsDiffEq};
use nalgebra::{
	base::allocator::Allocator,
	base::dimension::{U1, U2, U3, U4, U5, U6},
	constraint::{
		AreMultipliable, DimEq, SameDimension, SameNumberOfColumns, SameNumberOfRows,
		ShapeConstraint,
	},
	storage::{Owned, RawStorage, RawStorageMut, Storage, StorageMut},
	DefaultAllocator, Dim, DimName, DimNameDiff, DimNameSub, Matrix, MatrixView, MatrixViewMut,
	OMatrix, OVector, Scalar, SimdRealField, Unit, VectorView,
};
use num_traits::{real::Real, sign::Signed};
use std::ops::{Add, Neg, Sub};

/// Extension for $n$-dimensional Lorentzian space $\R^{-,+} = \R^{1,n-1}$ with
/// metric signature in spacelike sign convention.
///
/// In four dimensions also known as Minkowski space $\R^{-,+} = \R^{1,3}$.
///
/// A statically sized column-major matrix whose `R` rows and `C` columns
/// coincide with degree-1/degree-2 tensor indices.
pub trait LorentzianN<N, R, C>
where
	N: Scalar,
	R: DimName,
	C: DimName,
{
	/// Lorentzian metric tensor $\eta_{\mu \nu}$:
	///
	/// $$
	/// \eta_{\mu \nu} = \begin{pmatrix}
	///   -1   &    0   &  \dots &    0   \\\\
	///    0   &    1   & \ddots & \vdots \\\\
	/// \vdots & \ddots & \ddots &    0   \\\\
	///    0   &  \dots &    0   &    1   \end{pmatrix}
	/// $$
	///
	/// Avoid matrix multiplication by preferring:
	///
	///   * [`Self::dual`], [`Self::r_dual`] or [`Self::c_dual`] and their in-place counterparts
	///   * [`Self::dual_mut`], [`Self::r_dual_mut`] or [`Self::c_dual_mut`].
	///
	/// The spacelike sign convention $\R^{-,+} = \R^{1,n-1}$ requires less negations than its
	/// timelike alternative $\R^{+,-} = \R^{1,n-1}$. In four dimensions or Minkowski space
	/// $\R^{-,+} = \R^{1,3}$ it requires:
	///
	///   * $n - 2 = 2$ less for degree-1 tensors, and
	///   * $n (n - 2) = 8$ less for one index of degree-2 tensors, but
	///   * $0$ less for two indices of degree-2 tensors.
	///
	/// Choosing the component order of $\R^{-,+} = \R^{1,n-1}$ over $\R^{+,-} = \R^{n-1,1}$
	/// identifies the time component of $x^\mu$ as $x^0$ in compliance with the convention of
	/// identifying spatial components $x^i$ with Latin alphabet indices starting from $i=1$.
	/// ```
	/// use approx::assert_ulps_eq;
	/// use nalgebra::{Matrix4, Vector4};
	/// use nalgebra_spacetime::LorentzianN;
	///
	/// let eta = Matrix4::<f64>::metric();
	/// let sc = Vector4::new(-1.0, 1.0, 1.0, 1.0);
	/// assert_ulps_eq!(eta, Matrix4::from_diagonal(&sc));
	///
	/// let x = Vector4::<f64>::new_random();
	/// assert_ulps_eq!(x.dual(), eta * x);
	///
	/// let f = Matrix4::<f64>::new_random();
	/// assert_ulps_eq!(f.dual(), eta * f * eta);
	/// assert_ulps_eq!(f.dual(), f.r_dual().c_dual());
	/// assert_ulps_eq!(f.dual(), f.c_dual().r_dual());
	/// assert_ulps_eq!(f.r_dual(), eta * f);
	/// assert_ulps_eq!(f.c_dual(), f * eta);
	/// ```
	#[must_use]
	fn metric() -> Self
	where
		ShapeConstraint: SameDimension<R, C>;

	/// Raises/Lowers *all* of its degree-1/degree-2 tensor indices.
	///
	/// Negates the appropriate components avoiding matrix multiplication.
	#[must_use]
	fn dual(&self) -> Self;

	/// Raises/Lowers its degree-1/degree-2 *row* tensor index.
	///
	/// Prefer [`Self::dual`] over `self.r_dual().c_dual()` to half negations.
	#[must_use]
	fn r_dual(&self) -> Self;

	/// Raises/Lowers its degree-1/degree-2 *column* tensor index.
	///
	/// Prefer [`Self::dual`] over `self.r_dual().c_dual()` to half negations.
	#[must_use]
	fn c_dual(&self) -> Self;

	/// Raises/Lowers *all* of its degree-1/degree-2 tensor indices *in-place*.
	///
	/// Negates the appropriate components avoiding matrix multiplication.
	fn dual_mut(&mut self);

	/// Raises/Lowers its degree-1/degree-2 *row* tensor index *in-place*.
	///
	/// Prefer [`Self::dual`] over `self.r_dual_mut().c_dual_mut()` to half
	/// negations.
	fn r_dual_mut(&mut self);

	/// Raises/Lowers its degree-1/degree-2 *column* tensor index *in-place*.
	///
	/// Prefer [`Self::dual`] over `self.r_dual_mut().c_dual_mut()` to half negations.
	fn c_dual_mut(&mut self);

	/// Lorentzian matrix multiplication of degree-1/degree-2 tensors.
	///
	/// Equals `self.c_dual() * rhs`, the metric contraction of its *column* index with `rhs`'s
	/// *row* index.
	#[must_use]
	fn contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> OMatrix<N, R, C2>
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: AreMultipliable<R, C, R2, C2>,
		DefaultAllocator: Allocator<N, R, C2>;

	/// Same as [`Self::contr`] but with transposed tensor indices.
	///
	/// Equals `self.r_dual().tr_mul(rhs)`, the metric contraction of its *transposed row* index
	/// with `rhs`'s *row* index.
	#[must_use]
	fn tr_contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> OMatrix<N, C, C2>
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: SameNumberOfRows<R, R2>,
		DefaultAllocator: Allocator<N, C, C2>;

	/// Lorentzian inner product of degree-1/degree-2 tensors.
	///
	/// Equals `self.dual().dot(rhs)`, the metric contraction of:
	///
	///   * one index for degree-1 tensors, and
	///   * two indices for degree-2 tensors.
	///
	/// Also known as:
	///
	///   * Minkowski inner product,
	///   * relativistic dot product,
	///   * Lorentz scalar, invariant under Lorentz transformations, or
	///   * spacetime interval between two events, see [`Self::interval`].
	#[must_use]
	fn scalar<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>;

	/// Same as [`Self::scalar`] but with transposed tensor indices.
	///
	/// Equals `self.dual().tr_dot(rhs)`.
	#[must_use]
	fn tr_scalar<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: DimEq<C, R2> + DimEq<R, C2>;

	/// Lorentzian norm of timelike degree-1/degree-2 tensors.
	///
	/// Equals `self.scalar(self).neg().sqrt()`.
	///
	/// If spacelike, returns *NaN* or panics if `N` doesn't support it.
	#[must_use]
	fn timelike_norm(&self) -> N;

	/// Lorentzian norm of spacelike degree-1/degree-2 tensors.
	///
	/// Equals `self.scalar(self).sqrt()`.
	///
	/// If timelike, returns *NaN* or panics if `N` doesn't support it.
	#[must_use]
	fn spacelike_norm(&self) -> N;

	/// Spacetime interval between two events and region of `self`'s light cone.
	///
	/// Same as [`Self::interval_fn`] but with [`AbsDiffEq::default_epsilon`] as in:
	///
	///   * `is_present = |time| abs_diff_eq!(time, N::zero())`, and
	///   * `is_lightlike = |interval| abs_diff_eq!(interval, N::zero())`.
	#[must_use]
	fn interval(&self, rhs: &Self) -> (N, LightCone)
	where
		N: AbsDiffEq,
		ShapeConstraint: DimEq<U1, C>;

	/// Spacetime interval between two events and region of `self`'s light cone.
	///
	/// Equals `(rhs - self).scalar(&(rhs - self))` where `self` is subtracted from `rhs` to depict
	/// `rhs` in `self`'s light cone.
	///
	/// Requires you to approximate when `N` equals `N::zero()` via:
	///
	///   * `is_present` for the time component of `rhs - self`, and
	///   * `is_lightlike` for the interval.
	///
	/// Their signs are only evaluated in the `false` branches of `is_present` and `is_lightlike`.
	///
	/// See [`Self::interval`] for using defaults and [`approx`] for further details.
	#[must_use]
	fn interval_fn<P, L>(&self, rhs: &Self, is_present: P, is_lightlike: L) -> (N, LightCone)
	where
		ShapeConstraint: DimEq<U1, C>,
		P: Fn(N) -> bool,
		L: Fn(N) -> bool;

	/// $
	/// \gdef \uk {\hat u \cdot \vec K}
	/// \gdef \Lmu {\Lambda^{\mu\'}\_{\phantom {\mu\'} \mu}}
	/// \gdef \Lnu {(\Lambda^T)\_\nu^{\phantom \nu \nu\'}}
	/// $
	/// Lorentz transformation $\Lmu(\hat u, \zeta)$ boosting degree-1/degree-2 tensors to inertial
	/// `frame` of reference.
	///
	/// $$
	/// \Lmu(\hat u, \zeta) = I - \sinh \zeta (\uk) + (\cosh \zeta - 1) (\uk)^2
	/// $$
	///
	/// Where $\uk$ is the generator of the boost along $\hat{u}$ with its spatial componentsi
	/// $(u^1, \dots, u^{n-1})$:
	///
	/// $$
	/// \uk = \begin{pmatrix}
	///    0    &   u^1  &  \dots & u^{n-1} \\\\
	///   u^1   &    0   &  \dots &    0    \\\\
	/// \vdots  & \vdots & \ddots & \vdots  \\\\
	/// u^{n-1} &    0   &  \dots &    0    \end{pmatrix}
	/// $$
	///
	/// Boosts degree-1 tensors by multiplying it from the left:
	///
	/// $$
	/// x^{\mu\'} = \Lmu x^\mu
	/// $$
	///
	/// Boosts degree-2 tensors by multiplying it from the left and its transpose (symmetric for
	/// pure boosts) from the right:
	///
	/// $$
	/// F^{\mu\' \nu\'} = \Lmu F^{\mu \nu} \Lnu
	/// $$
	///
	/// ```
	/// use approx::assert_ulps_eq;
	/// use nalgebra::{Matrix4, Vector3, Vector4};
	/// use nalgebra_spacetime::{Frame4, LorentzianN};
	///
	/// let event = Vector4::new_random();
	/// let frame = Frame4::from_beta(Vector3::new(0.3, -0.4, 0.6));
	/// let boost = Matrix4::new_boost(&frame);
	/// assert_ulps_eq!(boost * event, event.boost(&frame), epsilon = 1e-14);
	/// ```
	#[must_use]
	fn new_boost<D>(frame: &FrameN<N, D>) -> Self
	where
		D: DimNameSub<U1>,
		ShapeConstraint: AreMultipliable<R, C, R, C> + DimEq<R, D>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>;

	/// Boosts this degree-1 tensor $x^\mu$ to inertial `frame` of reference along $\hat u$ with
	/// $\zeta$.
	///
	/// ```
	/// use approx::assert_ulps_eq;
	/// use nalgebra::{Vector3, Vector4};
	/// use nalgebra_spacetime::{Frame4, LorentzianN};
	///
	/// let muon_lifetime_at_rest = Vector4::new(2.2e-6, 0.0, 0.0, 0.0);
	/// let muon_frame = Frame4::from_axis_beta(Vector3::z_axis(), 0.9952);
	/// let muon_lifetime = muon_lifetime_at_rest.boost(&muon_frame);
	/// let time_dilation_factor = muon_lifetime[0] / muon_lifetime_at_rest[0];
	/// assert_ulps_eq!(time_dilation_factor, 10.218, epsilon = 1e-3);
	/// ```
	///
	/// See `boost_mut()` for further details.
	#[must_use]
	fn boost<D>(&self, frame: &FrameN<N, D>) -> Self
	where
		R: DimNameSub<U1>,
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D>
			+ SameNumberOfColumns<C, U1>
			+ DimEq<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>
			+ SameNumberOfRows<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>;

	/// $
	/// \gdef \xu {(\vec x \cdot \hat u)}
	/// $
	/// Boosts this degree-1 tensor $x^\mu$ to inertial `frame` of reference along $\hat u$ with
	/// $\zeta$ *in-place*.
	///
	/// $$
	/// \begin{pmatrix}
	/// x^0 \\\\
	/// \vec x
	/// \end{pmatrix}\' = \begin{pmatrix}
	/// x^0 \cosh \zeta - \xu \sinh \zeta \\\\
	/// \vec x + (\xu (\cosh \zeta - 1) - x^0 \sinh \zeta) \hat u
	/// \end{pmatrix}
	/// $$
	///
	/// ```
	/// use approx::assert_ulps_eq;
	/// use nalgebra::Vector4;
	/// use nalgebra_spacetime::LorentzianN;
	///
	/// // Arbitrary timelike four-momentum.
	/// let mut momentum = Vector4::new(24.3, 5.22, 16.8, 9.35);
	///
	/// // Rest mass.
	/// let mass = momentum.timelike_norm();
	/// // Four-momentum in center-of-momentum frame.
	/// let mass_at_rest = Vector4::new(mass, 0.0, 0.0, 0.0);
	///
	/// // Rest mass is ratio of four-momentum to four-velocity.
	/// let velocity = momentum / mass;
	/// // Four-momentum boosted to center-of-momentum frame.
	/// momentum.boost_mut(&velocity.frame());
	///
	/// // Verify boosting four-momentum to center-of-momentum frame.
	/// assert_ulps_eq!(momentum, mass_at_rest, epsilon = 1e-14);
	/// ```
	fn boost_mut<D>(&mut self, frame: &FrameN<N, D>)
	where
		R: DimNameSub<U1>,
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D>
			+ SameNumberOfColumns<C, U1>
			+ DimEq<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>
			+ SameNumberOfRows<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>;

	/// Velocity $u^\mu$ of inertial `frame` of reference.
	#[must_use]
	fn new_velocity<D>(frame: &FrameN<N, D>) -> OVector<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>;

	/// Inertial frame of reference of this velocity $u^\mu$.
	#[must_use]
	fn frame(&self) -> FrameN<N, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<R, U1>>;

	/// From `temporal` and `spatial` spacetime split.
	#[must_use]
	fn from_split<D>(temporal: &N, spatial: &OVector<N, DimNameDiff<D, U1>>) -> OVector<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			StorageMut<N, D, U1, RStride = U1, CStride = D>;

	/// Spacetime split into [`Self::temporal`] and [`Self::spatial`].
	#[must_use]
	fn split(&self) -> (&N, MatrixView<N, DimNameDiff<R, U1>, C, U1, R>)
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>;

	/// Mutable spacetime split into [`Self::temporal_mut`] and
	/// [`Self::spatial_mut`].
	///
	/// ```
	/// use approx::assert_ulps_eq;
	/// use nalgebra::Vector4;
	/// use nalgebra_spacetime::LorentzianN;
	///
	/// let mut spacetime = Vector4::new(1.0, 2.0, 3.0, 4.0);
	/// let (temporal, mut spatial) = spacetime.split_mut();
	/// *temporal += 1.0;
	/// spatial[0] += 2.0;
	/// spatial[1] += 3.0;
	/// spatial[2] += 4.0;
	/// assert_ulps_eq!(spacetime, Vector4::new(2.0, 4.0, 6.0, 8.0));
	/// ```
	fn split_mut(&mut self) -> (&mut N, MatrixViewMut<N, DimNameDiff<R, U1>, C>)
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>;

	/// Temporal component.
	#[must_use]
	fn temporal(&self) -> &N
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>;

	/// Mutable temporal component.
	fn temporal_mut(&mut self) -> &mut N
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>;

	/// Spatial components.
	#[must_use]
	fn spatial(&self) -> MatrixView<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>;

	/// Mutable spatial components.
	fn spatial_mut(&mut self) -> MatrixViewMut<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			StorageMut<N, R, C, RStride = U1, CStride = R>;
}

impl<N, R, C> LorentzianN<N, R, C> for OMatrix<N, R, C>
where
	N: SimdRealField + Signed + Real,
	R: DimName + DimNameSub<U1>,
	C: DimName,
	DefaultAllocator: Allocator<N, R, C>,
{
	#[inline]
	fn metric() -> Self
	where
		ShapeConstraint: SameDimension<R, C>,
	{
		let mut m = Self::identity();
		neg(m.get_mut((0, 0)).unwrap());
		m
	}

	#[inline]
	fn dual(&self) -> Self {
		let mut m = self.clone_owned();
		m.dual_mut();
		m
	}

	#[inline]
	fn r_dual(&self) -> Self {
		let mut m = self.clone_owned();
		m.r_dual_mut();
		m
	}

	#[inline]
	fn c_dual(&self) -> Self {
		let mut m = self.clone_owned();
		m.c_dual_mut();
		m
	}

	#[inline]
	fn dual_mut(&mut self) {
		if R::is::<U1>() || C::is::<U1>() {
			neg(self.get_mut(0).unwrap());
		} else if R::is::<C>() {
			for i in 1..R::dim() {
				neg(self.get_mut((i, 0)).unwrap());
				neg(self.get_mut((0, i)).unwrap());
			}
		} else {
			for r in 1..R::dim() {
				neg(self.get_mut((r, 0)).unwrap());
			}
			for c in 1..C::dim() {
				neg(self.get_mut((0, c)).unwrap());
			}
		}
	}

	#[inline]
	fn r_dual_mut(&mut self) {
		for c in 0..C::dim() {
			neg(self.get_mut((0, c)).unwrap());
		}
	}

	#[inline]
	fn c_dual_mut(&mut self) {
		for r in 0..R::dim() {
			neg(self.get_mut((r, 0)).unwrap());
		}
	}

	#[inline]
	fn contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> OMatrix<N, R, C2>
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: AreMultipliable<R, C, R2, C2>,
		DefaultAllocator: Allocator<N, R, C2>,
	{
		self.c_dual() * rhs
	}

	#[inline]
	fn tr_contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> OMatrix<N, C, C2>
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: SameNumberOfRows<R, R2>,
		DefaultAllocator: Allocator<N, C, C2>,
	{
		self.r_dual().tr_mul(rhs)
	}

	#[inline]
	fn scalar<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
	{
		self.dual().dot(rhs)
	}

	#[inline]
	fn tr_scalar<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: DimEq<C, R2> + DimEq<R, C2>,
	{
		self.dual().tr_dot(rhs)
	}

	#[inline]
	fn timelike_norm(&self) -> N {
		self.scalar(self).neg().sqrt()
	}

	#[inline]
	fn spacelike_norm(&self) -> N {
		self.scalar(self).sqrt()
	}

	#[inline]
	fn interval(&self, rhs: &Self) -> (N, LightCone)
	where
		N: AbsDiffEq,
		ShapeConstraint: DimEq<U1, C>,
	{
		self.interval_fn(
			rhs,
			|time| abs_diff_eq!(time, N::zero()),
			|interval| abs_diff_eq!(interval, N::zero()),
		)
	}

	fn interval_fn<P, L>(&self, rhs: &Self, is_present: P, is_lightlike: L) -> (N, LightCone)
	where
		ShapeConstraint: DimEq<U1, C>,
		P: Fn(N) -> bool,
		L: Fn(N) -> bool,
	{
		let time = self[0];
		let difference = rhs - self;
		let interval = difference.scalar(&difference);
		let light_cone = if is_lightlike(interval) {
			if is_present(time) {
				LightCone::Origin
			} else if time.is_sign_positive() {
				LightCone::LightlikeFuture
			} else {
				LightCone::LightlikePast
			}
		} else if interval.is_sign_positive() || is_present(time) {
			LightCone::Spacelike
		} else if time.is_sign_positive() {
			LightCone::TimelikeFuture
		} else {
			LightCone::TimelikePast
		};
		(interval, light_cone)
	}

	fn new_boost<D>(frame: &FrameN<N, D>) -> Self
	where
		D: DimNameSub<U1>,
		ShapeConstraint: AreMultipliable<R, C, R, C> + DimEq<R, D>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let &FrameN {
			zeta_cosh,
			zeta_sinh,
			ref axis,
		} = frame;
		let mut b = Self::zeros();
		for (i, u) in axis.iter().enumerate() {
			b[(i + 1, 0)] = *u;
			b[(0, i + 1)] = *u;
		}
		let uk = b.clone_owned();
		b.gemm(zeta_cosh - N::one(), &uk, &uk, -zeta_sinh);
		for i in 0..D::dim() {
			b[(i, i)] += N::one();
		}
		b
	}

	#[inline]
	fn boost<D>(&self, frame: &FrameN<N, D>) -> Self
	where
		R: DimNameSub<U1>,
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D>
			+ SameNumberOfColumns<C, U1>
			+ DimEq<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>
			+ SameNumberOfRows<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let mut v = self.clone_owned();
		v.boost_mut(frame);
		v
	}

	fn boost_mut<D>(&mut self, frame: &FrameN<N, D>)
	where
		R: DimNameSub<U1>,
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D>
			+ SameNumberOfColumns<C, U1>
			+ DimEq<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>
			+ SameNumberOfRows<<R as DimNameSub<U1>>::Output, <D as DimNameSub<U1>>::Output>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let &FrameN {
			zeta_cosh,
			zeta_sinh,
			ref axis,
		} = frame;
		let u = axis.as_ref();
		let a = self[0];
		let (rows, _cols) = self.shape_generic();
		let zu = self.rows_generic(1, rows.sub(U1)).dot(u);
		self[0] = zeta_cosh * a - zeta_sinh * zu;
		let mut z = self.rows_generic_mut(1, rows.sub(U1));
		z += u * ((zeta_cosh - N::one()) * zu - zeta_sinh * a);
	}

	#[inline]
	fn new_velocity<D>(frame: &FrameN<N, D>) -> OVector<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
	{
		frame.velocity()
	}

	#[inline]
	fn frame(&self) -> FrameN<N, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<R, U1>>,
	{
		FrameN::<N, R>::from_velocity(self)
	}

	#[inline]
	fn from_split<D>(temporal: &N, spatial: &OVector<N, DimNameDiff<D, U1>>) -> OVector<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			StorageMut<N, D, U1, RStride = U1, CStride = D>,
	{
		let mut v = OVector::<N, D>::zeros();
		*v.temporal_mut() = *temporal;
		v.spatial_mut().copy_from(spatial);
		v
	}

	#[inline]
	fn split(&self) -> (&N, MatrixView<N, DimNameDiff<R, U1>, C, U1, R>)
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>,
	{
		(self.temporal(), self.spatial())
	}

	#[inline]
	fn split_mut(&mut self) -> (&mut N, MatrixViewMut<N, DimNameDiff<R, U1>, C>)
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>,
	{
		let (temporal, spatial) = self.as_mut_slice().split_at_mut(1);
		(
			temporal.get_mut(0).unwrap(),
			MatrixViewMut::<N, DimNameDiff<R, U1>, C>::from_slice(spatial),
		)
	}

	#[inline]
	fn temporal(&self) -> &N
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
	{
		self.get(0).unwrap()
	}

	#[inline]
	fn temporal_mut(&mut self) -> &mut N
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
	{
		self.get_mut(0).unwrap()
	}

	#[inline]
	fn spatial(&self) -> MatrixView<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			RawStorage<N, R, C, RStride = U1, CStride = R>,
	{
		let (rows, _cols) = self.shape_generic();
		self.rows_generic(1, rows.sub(U1))
	}

	#[inline]
	fn spatial_mut(&mut self) -> MatrixViewMut<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			RawStorageMut<N, R, C, RStride = U1, CStride = R>,
	{
		let (rows, _cols) = self.shape_generic();
		self.rows_generic_mut(1, rows.sub(U1))
	}
}

#[inline]
fn neg<N: Scalar + Signed>(n: &mut N) {
	*n = -n.clone();
}

/// Spacetime regions regarding an event's light cone.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LightCone {
	/// *Interior* events of the future-directed light cone.
	TimelikeFuture,
	/// Events *on* the future-directed light cone itself.
	LightlikeFuture,
	/// Trivial *zero-vector* difference of two coinciding events.
	Origin,
	/// Events *on* the past-directed light cone itself.
	LightlikePast,
	/// *Interior* events of the past-directed light cone.
	TimelikePast,
	/// Events *elsewhere*.
	Spacelike,
}

/// Inertial frame of reference in $2$-dimensional Lorentzian space
/// $\R^{-,+} = \R^{1,1}$.
pub type Frame2<N> = FrameN<N, U2>;
/// Inertial frame of reference in $3$-dimensional Lorentzian space
/// $\R^{-,+} = \R^{1,2}$.
pub type Frame3<N> = FrameN<N, U3>;
/// Inertial frame of reference in $4$-dimensional Lorentzian space
/// $\R^{-,+} = \R^{1,3}$.
pub type Frame4<N> = FrameN<N, U4>;
/// Inertial frame of reference in $5$-dimensional Lorentzian space
/// $\R^{-,+} = \R^{1,4}$.
pub type Frame5<N> = FrameN<N, U5>;
/// Inertial frame of reference in $6$-dimensional Lorentzian space
/// $\R^{-,+} = \R^{1,5}$.
pub type Frame6<N> = FrameN<N, U6>;

/// Inertial frame of reference in $n$-dimensional Lorentzian space $\R^{-,+} = \R^{1,n-1}$.
///
/// Holds a statically sized direction axis $\hat u \in \R^{n-1}$ and two boost parameters
/// precomputed from either velocity $u^\mu$, rapidity $\vec \zeta$, or velocity ratio $\vec \beta$
/// whether using [`Self::from_velocity`], [`Self::from_zeta`], or [`Self::from_beta`]:
///
/// $$
/// \cosh \zeta = \gamma
/// $$
///
/// $$
/// \sinh \zeta = \beta \gamma
/// $$
///
/// Where $\gamma$ is the Lorentz factor.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FrameN<N, D>
where
	N: Scalar,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
{
	zeta_cosh: N,
	zeta_sinh: N,
	axis: Unit<OVector<N, DimNameDiff<D, U1>>>,
}

impl<N, D> FrameN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
{
	/// Inertial frame of reference with velocity $u^\mu$.
	#[must_use]
	pub fn from_velocity<R, C>(u: &OMatrix<N, R, C>) -> Self
	where
		R: DimNameSub<U1>,
		C: Dim,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, R, C>,
	{
		let mut scaled_axis = OVector::zeros();
		let zeta_cosh = u[0];
		let (rows, _cols) = u.shape_generic();
		scaled_axis
			.iter_mut()
			.zip(u.rows_generic(1, rows.sub(U1)).iter())
			.for_each(|(scaled_axis, &u)| *scaled_axis = u);
		let (axis, zeta_sinh) = Unit::new_and_get(scaled_axis);
		Self {
			zeta_cosh,
			zeta_sinh,
			axis,
		}
	}

	/// Inertial frame of reference with rapidity $\vec \zeta$.
	#[must_use]
	#[inline]
	pub fn from_zeta(scaled_axis: OVector<N, DimNameDiff<D, U1>>) -> Self {
		let (axis, zeta) = Unit::new_and_get(scaled_axis);
		Self::from_axis_zeta(axis, zeta)
	}

	/// Inertial frame of reference along $\hat u$ with rapidity $\zeta$.
	#[must_use]
	#[inline]
	pub fn from_axis_zeta(axis: Unit<OVector<N, DimNameDiff<D, U1>>>, zeta: N) -> Self {
		Self {
			zeta_cosh: zeta.cosh(),
			zeta_sinh: zeta.sinh(),
			axis,
		}
	}

	/// Inertial frame of reference with velocity ratio $\vec \beta$.
	#[must_use]
	#[inline]
	pub fn from_beta(scaled_axis: OVector<N, DimNameDiff<D, U1>>) -> Self {
		let (axis, beta) = Unit::new_and_get(scaled_axis);
		Self::from_axis_beta(axis, beta)
	}

	/// Inertial frame of reference along $\hat u$ with velocity ratio $\beta$.
	#[must_use]
	#[inline]
	pub fn from_axis_beta(axis: Unit<OVector<N, DimNameDiff<D, U1>>>, beta: N) -> Self {
		debug_assert!(
			-N::one() < beta && beta < N::one(),
			"Velocity ratio `beta` is out of range (-1, +1)"
		);
		let gamma = N::one() / (N::one() - beta * beta).sqrt();
		Self {
			zeta_cosh: gamma,
			zeta_sinh: beta * gamma,
			axis,
		}
	}

	/// Velocity $u^\mu$.
	#[must_use]
	pub fn velocity(&self) -> OVector<N, D>
	where
		DefaultAllocator: Allocator<N, D>,
	{
		let mut u = OVector::<N, D>::zeros();
		u[0] = self.gamma();
		let (rows, _cols) = u.shape_generic();
		u.rows_generic_mut(1, rows.sub(U1))
			.iter_mut()
			.zip(self.axis.iter())
			.for_each(|(u, &axis)| *u = self.beta_gamma() * axis);
		u
	}

	/// Direction $\hat u$.
	#[must_use]
	#[inline]
	pub fn axis(&self) -> Unit<OVector<N, DimNameDiff<D, U1>>> {
		self.axis.clone()
	}

	/// Rapidity $\zeta$.
	#[must_use]
	#[inline]
	pub fn zeta(&self) -> N {
		self.beta().atanh()
	}

	/// Velocity ratio $\beta$.
	#[must_use]
	#[inline]
	pub fn beta(&self) -> N {
		self.beta_gamma() / self.gamma()
	}

	/// Lorentz factor $\gamma$.
	#[must_use]
	#[inline]
	pub const fn gamma(&self) -> N {
		self.zeta_cosh
	}

	/// Product of velocity ratio $\beta$ and Lorentz factor $\gamma$.
	#[must_use]
	#[inline]
	pub const fn beta_gamma(&self) -> N {
		self.zeta_sinh
	}

	/// Relativistic velocity addition `self`$\oplus$`frame`.
	///
	/// Equals `frame.velocity().boost(&-self.clone()).frame()`.
	#[must_use]
	#[inline]
	pub fn compose(&self, frame: &Self) -> Self
	where
		DefaultAllocator: Allocator<N, D>,
	{
		frame.velocity().boost(&-self.clone()).frame()
	}

	/// Wigner rotation angle $\epsilon$ of the boost composition `self`$\oplus$`frame`.
	///
	/// The angle between the forward and backward composition:
	///
	/// $$
	/// \epsilon = \angle (\vec \beta_u \oplus \vec \beta_v, \vec \beta_v \oplus \vec \beta_u)
	/// $$
	///
	/// See [`Self::rotation`] for further details.
	#[must_use]
	pub fn angle(&self, frame: &Self) -> N
	where
		DefaultAllocator: Allocator<N, D>,
	{
		let (u, v) = (self, frame);
		let ucv = u.compose(v);
		let vcu = v.compose(u);
		ucv.axis().angle(&vcu.axis())
	}

	/// $
	/// \gdef \Bu {B^{\mu\'}\_{\phantom {\mu\'} \mu} (\vec \beta_u)}
	/// \gdef \Bv {B^{\mu\'\'}\_{\phantom {\mu\'\'} \mu\'} (\vec \beta_v)}
	/// \gdef \Puv {u \oplus v}
	/// \gdef \Buv {B^{\mu\'}\_{\phantom {\mu\'} \mu} (\vec \beta_{\Puv})}
	/// \gdef \Ruv {R^{\mu\'\'}\_{\phantom {\mu\'\'} \mu\'} (\epsilon)}
	/// \gdef \R {R (\epsilon)}
	/// \gdef \Kuv {K(\epsilon)}
	/// \gdef \Luv {\Lambda^{\mu\'\'}\_{\phantom {\mu\'\'} \mu} (\vec \beta_{\Puv})}
	/// $
	/// Wigner rotation matrix $R(\widehat {\vec \beta_u \wedge \vec \beta_v}, \epsilon)$ of the
	/// boost composition `self`$\oplus$`frame`.
	///
	/// The composition of two pure boosts, $\Bu$ to `self` followed by $\Bv$ to `frame`, results in
	/// a composition of a pure boost $\Buv$ and a *non-vanishing* spatial rotation $\Ruv$ for
	/// *non-collinear* boosts:
	///
	/// $$
	/// \Luv = \Ruv \Buv = \Bv \Bu
	/// $$
	///
	/// The returned homogeneous rotation matrix
	///
	/// $$
	/// \R = \begin{pmatrix}
	///      1      & \vec{0}^T \\\\
	///   \vec{0}   &   \Kuv    \end{pmatrix}
	/// $$
	///
	/// embeds the spatial rotation matrix
	///
	/// $$
	/// \Kuv = I + \sin \epsilon (\hat v \hat u^T - \hat u \hat v^T)
	///          + (\cos \epsilon - 1) (\hat u \hat u^T + \hat v \hat v^T)
	/// $$
	///
	/// rotating in the plane spanned by the orthonormalized pair $\hat u = \hat \beta_u$ and
	/// $\hat v = \widehat{\hat \beta_v - (\hat \beta_u \cdot \hat \beta_v) \hat \beta_u}$. The
	/// rotation angle $\epsilon$ is found according to [`Self::angle`].
	///
	/// See [`Self::axis_angle`] for $4$-dimensional specialization.
	///
	/// ```
	/// use approx::{assert_ulps_eq, assert_ulps_ne};
	/// use nalgebra::{Matrix4, Vector3};
	/// use nalgebra_spacetime::{Frame4, LorentzianN};
	///
	/// let u = Frame4::from_beta(Vector3::new(0.18, 0.73, 0.07));
	/// let v = Frame4::from_beta(Vector3::new(0.41, 0.14, 0.25));
	/// let ucv = u.compose(&v);
	/// let vcu = v.compose(&u);
	///
	/// let boost_u = Matrix4::new_boost(&u);
	/// let boost_v = Matrix4::new_boost(&v);
	/// let boost_ucv = Matrix4::new_boost(&ucv);
	/// let boost_vcu = Matrix4::new_boost(&vcu);
	///
	/// let (matrix_ucv, angle_ucv) = u.rotation(&v);
	///
	/// assert_ulps_eq!(angle_ucv, u.angle(&v));
	/// assert_ulps_ne!(boost_ucv, boost_v * boost_u);
	/// assert_ulps_ne!(boost_vcu, boost_u * boost_v);
	/// assert_ulps_eq!(matrix_ucv * boost_ucv, boost_v * boost_u);
	/// assert_ulps_eq!(boost_vcu * matrix_ucv, boost_v * boost_u);
	/// ```
	#[must_use]
	pub fn rotation(&self, frame: &Self) -> (OMatrix<N, D, D>, N)
	where
		DefaultAllocator: Allocator<N, D>
			+ Allocator<N, D, D>
			+ Allocator<N, U1, DimNameDiff<D, U1>>
			+ Allocator<N, DimNameDiff<D, U1>, DimNameDiff<D, U1>>,
	{
		let [u, v] = [self, frame];
		let ang = u.angle(v);
		let (sin, cos) = ang.sin_cos();
		let u = u.axis().into_inner();
		let v = v.axis().into_inner();
		let v = (&v - &u * u.dot(&v)).normalize();
		let ut = u.transpose();
		let vt = v.transpose();
		let rot = (&v * &ut - &u * &vt) * sin + (&u * &ut + &v * &vt) * (cos - N::one());
		let mut mat = OMatrix::<N, D, D>::identity();
		let (r, c) = mat.shape_generic();
		let mut sub = mat.generic_view_mut((1, 1), (r.sub(U1), c.sub(U1)));
		sub += rot;
		(mat, ang)
	}

	/// Wigner rotation axis $\widehat {\vec \beta_u \times \vec \beta_v}$ and angle $\epsilon$ of
	/// the boost composition `self`$\oplus$`frame`.
	///
	/// $$
	/// \epsilon = \arcsin \Bigg({
	///   1 + \gamma + \gamma_u + \gamma_v
	///   \over
	///   (1 + \gamma) (1 + \gamma_u) (1 + \gamma_v)
	/// } \gamma_u \gamma_v \|\vec \beta_u \times \vec \beta_v\| \Bigg)
	/// $$
	///
	/// $$
	/// \gamma = \gamma_u \gamma_v (1 + \vec \beta_u \cdot \vec \beta_v)
	/// $$
	///
	/// See [`Self::rotation`] for $n$-dimensional generalization.
	///
	/// ```
	/// use approx::{assert_abs_diff_ne, assert_ulps_eq};
	/// use nalgebra::Vector3;
	/// use nalgebra_spacetime::{Frame4, LorentzianN};
	///
	/// let u = Frame4::from_beta(Vector3::new(0.18, 0.73, 0.07));
	/// let v = Frame4::from_beta(Vector3::new(0.41, 0.14, 0.25));
	///
	/// let ucv = u.compose(&v).axis();
	/// let vcu = v.compose(&u).axis();
	///
	/// let (axis, angle) = u.axis_angle(&v);
	/// let axis = axis.into_inner();
	///
	/// assert_abs_diff_ne!(angle, 0.0, epsilon = 1e-15);
	/// assert_ulps_eq!(angle, ucv.angle(&vcu), epsilon = 1e-15);
	/// assert_ulps_eq!(axis, ucv.cross(&vcu).normalize(), epsilon = 1e-15);
	/// ```
	#[must_use]
	pub fn axis_angle(&self, frame: &Self) -> (Unit<OVector<N, DimNameDiff<D, U1>>>, N)
	where
		N: SimdRealField + Signed + Real,
		D: DimNameSub<U1>,
		ShapeConstraint: DimEq<D, U4>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let (u, v) = (self, frame);
		let (axis, sin) = Unit::new_and_get(u.axis().cross(&v.axis()));
		let uv = u.axis().dot(&v.axis());
		let bg = u.beta_gamma() * v.beta_gamma();
		let ug = u.gamma();
		let vg = v.gamma();
		let cg = ug * vg + bg * uv;
		let sum = N::one() + cg + ug + vg;
		let prod = (N::one() + cg) * (N::one() + ug) * (N::one() + vg);
		(axis, (sum / prod * bg * sin).asin())
	}
}

impl<N, R, C> From<OMatrix<N, R, C>> for FrameN<N, R>
where
	N: SimdRealField + Signed + Real,
	R: DimNameSub<U1>,
	C: DimName,
	ShapeConstraint: SameNumberOfColumns<C, U1>,
	DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimNameDiff<R, U1>>,
{
	#[inline]
	fn from(u: OMatrix<N, R, C>) -> Self {
		u.frame()
	}
}

impl<N, D> From<FrameN<N, D>> for OVector<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
{
	#[inline]
	fn from(frame: FrameN<N, D>) -> Self {
		frame.velocity()
	}
}

impl<N, D> Neg for FrameN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
{
	type Output = Self;

	#[inline]
	fn neg(mut self) -> Self::Output {
		self.axis = -self.axis;
		self
	}
}

impl<N, D> Add for FrameN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
{
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self::Output {
		self.compose(&rhs)
	}
}

impl<N, D> Sub for FrameN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
{
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self::Output {
		self.compose(&-rhs)
	}
}

impl<N, D> Copy for FrameN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	Owned<N, DimNameDiff<D, U1>>: Copy,
{
}

/// Momentum in $n$-dimensional Lorentzian space $\R^{-,+} = \R^{1,n-1}$.
///
/// Assuming unit system with speed of light $c=1$ and rest mass $m$ as timelike norm in spacelike
/// sign convention as in:
///
/// $$
/// m^2=E^2-\vec {p}^2=-p_\mu p^\mu
/// $$
///
/// Where $p^\mu$ is the $n$-momentum with energy $E$ as temporal $p^0$ and momentum $\vec p$ as
/// spatial $p^i$ components:
///
/// $$
/// p^\mu = m u^\mu = m \begin{pmatrix}
///   \gamma \\\\ \gamma \vec \beta
/// \end{pmatrix} = \begin{pmatrix}
///   \gamma m = E \\\\ \gamma m \vec \beta = \vec p
/// \end{pmatrix}
/// $$
///
/// With $n$-velocity $u^\mu$, Lorentz factor $\gamma$, and velocity ratio $\vec \beta$.
#[derive(Debug, PartialEq, Clone)]
pub struct MomentumN<N, D>
where
	N: Scalar,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	momentum: OVector<N, D>,
}

impl<N, D> MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	/// Momentum with spacetime [`LorentzianN::split`], `energy` $E$ and
	/// `momentum` $\vec p$.
	#[must_use]
	#[inline]
	pub fn from_split(energy: &N, momentum: &OVector<N, DimNameDiff<D, U1>>) -> Self
	where
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			StorageMut<N, D, U1, RStride = U1, CStride = D>,
	{
		Self {
			momentum: OVector::<N, D>::from_split(energy, momentum),
		}
	}

	/// Momentum $p^\mu=m u^\mu$ with rest `mass` $m$ at `velocity` $u^\mu$.
	#[must_use]
	#[inline]
	pub fn from_mass_at_velocity(mass: N, velocity: OVector<N, D>) -> Self
	where
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		Self {
			momentum: velocity * mass,
		}
	}

	/// Momentum $p^\mu$ with rest `mass` $m$ in `frame`.
	///
	/// Equals `frame.velocity() * mass`.
	#[must_use]
	#[inline]
	pub fn from_mass_in_frame(mass: N, frame: &FrameN<N, D>) -> Self
	where
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		Self::from_mass_at_velocity(mass, frame.velocity())
	}

	/// Momentum $p^\mu$ with rest `mass` $m$ in center-of-momentum frame.
	#[must_use]
	#[inline]
	pub fn from_mass_at_rest(mass: N) -> Self {
		let mut momentum = OVector::<N, D>::zeros();
		*momentum.temporal_mut() = mass;
		Self { momentum }
	}

	/// Rest mass $m$ as timelike norm $\sqrt{-p_\mu p^\mu}$ in spacelike sign convention.
	#[must_use]
	#[inline]
	pub fn mass(&self) -> N {
		self.momentum.timelike_norm()
	}

	/// Velocity $u^\mu$ as momentum $p^\mu$ divided by rest `mass()` $m$.
	#[must_use]
	#[inline]
	pub fn velocity(&self) -> OVector<N, D> {
		self.momentum.clone() / self.mass()
	}

	/// Energy $E$ as [`LorentzianN::temporal`] component.
	#[must_use]
	#[inline]
	pub fn energy(&self) -> &N {
		self.momentum.temporal()
	}

	/// Momentum $\vec p$ as [`LorentzianN::spatial`] components.
	#[must_use]
	#[inline]
	pub fn momentum(&self) -> VectorView<N, DimNameDiff<D, U1>, U1, D>
	where
		DefaultAllocator: Allocator<N, D, U1>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			Storage<N, D, U1, RStride = U1, CStride = D>,
	{
		self.momentum.spatial()
	}
}

impl<N, D> From<OVector<N, D>> for MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	#[inline]
	fn from(momentum: OVector<N, D>) -> Self {
		Self { momentum }
	}
}

impl<N, D> From<MomentumN<N, D>> for OVector<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	#[inline]
	fn from(momentum: MomentumN<N, D>) -> Self {
		momentum.momentum
	}
}

impl<N, D> Neg for MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	type Output = Self;

	#[inline]
	fn neg(mut self) -> Self::Output {
		self.momentum = -self.momentum;
		self
	}
}

impl<N, D> Add<Self> for MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self::Output {
		Self {
			momentum: self.momentum + rhs.momentum,
		}
	}
}

impl<N, D> Sub<Self> for MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self::Output {
		Self {
			momentum: self.momentum - rhs.momentum,
		}
	}
}

impl<N, D> Copy for MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
	Owned<N, D>: Copy,
{
}

/// Momentum in $2$-dimensional Lorentzian space $\R^{-,+} = \R^{1,1}$.
pub type Momentum2<N> = MomentumN<N, U2>;
/// Momentum in $3$-dimensional Lorentzian space $\R^{-,+} = \R^{1,2}$.
pub type Momentum3<N> = MomentumN<N, U3>;
/// Momentum in $4$-dimensional Lorentzian space $\R^{-,+} = \R^{1,3}$.
pub type Momentum4<N> = MomentumN<N, U4>;
/// Momentum in $5$-dimensional Lorentzian space $\R^{-,+} = \R^{1,4}$.
pub type Momentum5<N> = MomentumN<N, U5>;
/// Momentum in $6$-dimensional Lorentzian space $\R^{-,+} = \R^{1,5}$.
pub type Momentum6<N> = MomentumN<N, U6>;
