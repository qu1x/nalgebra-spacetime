//! Spacetime Extension for [nalgebra]
//!
//! [nalgebra]: https://nalgebra.org
//!
//! # Present Features
//!
//!   * Minkowski space as special case of `LorentzianMN` space.
//!   * Raising/Lowering tensor indices: `dual()`/`r_dual()`/`c_dual()`.
//!   * Metric contraction of degree-1/degree-2 tensors: `contr()`/`scalar()`.
//!   * Spacetime `interval()` with `LightCone` depiction.
//!   * Inertial `FrameN` of reference holding boost parameters.
//!   * Lorentz boost as `new_boost()` matrix.
//!   * Direct Lorentz `boost()` to `compose()` velocities.
//!   * Wigner `rotation()` and `axis_angle()` between to-be-composed boosts.
//!
//! # Future Features
//!
//!   * `Event4`/`Velocity4`/`Momentum4`/`...` equivalents of `Point4`/`...`.
//!   * Categorize `Rotation4`/`PureBoost4`/`...` as `Boost4`/`...`.
//!   * Wigner `rotation()` and `axis_angle()` of an already-composed `Boost4`.
//!   * Distinguish pre/post-rotation and active/passive `Boost4` compositions.
//!   * Spacetime algebra (STA) as special case of `CliffordMN` space.

#![deny(missing_docs)]

use std::ops::{Neg, Add, Sub};
use nalgebra::{
	Unit, VectorN, Vector3, Matrix, MatrixMN, Matrix4, Rotation3,
	VectorSliceN, MatrixSliceMN, MatrixSliceMutMN,
	Scalar, SimdRealField,
	Dim, DimName, DimNameSub, DimNameDiff,
	base::dimension::*,
	constraint::{
		ShapeConstraint,
		SameDimension,
		SameNumberOfRows,
		SameNumberOfColumns,
		AreMultipliable,
		DimEq,
	},
	storage::{Storage, StorageMut, Owned},
	DefaultAllocator, base::allocator::Allocator,
};
use num_traits::{sign::Signed, real::Real};
use approx::{AbsDiffEq, abs_diff_eq};
use LightCone::*;

/// Extension for $n$-dimensional Lorentzian space $\R^{-,+} = \R^{1,n-1}$ with
/// metric signature in spacelike sign convention.
///
/// In four dimensions also known as Minkowski space $\R^{-,+} = \R^{1,3}$.
///
/// A statically sized column-major matrix whose `R` rows and `C` columns
/// coincide with degree-1/degree-2 tensor indices.
pub trait LorentzianMN<N, R, C>
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
	///   * `dual()`, `r_dual()` or `c_dual()` and their in-place counterparts
	///   * `dual_mut()`, `r_dual_mut()` or `c_dual_mut()`.
	///
	/// The spacelike sign convention $\R^{-,+} = \R^{1,n-1}$ requires less
	/// negations than its timelike alternative $\R^{+,-} = \R^{1,n-1}$. In four
	/// dimensions or Minkowski space $\R^{-,+} = \R^{1,3}$ it requires:
	///
	///   * $n - 2 = 2$ less for degree-1 tensors, and
	///   * $n (n - 2) = 8$ less for one index of degree-2 tensors, but
	///   * $0$ less for two indices of degree-2 tensors.
	///
	/// Choosing the component order of $\R^{-,+} = \R^{1,n-1}$ over
	/// $\R^{+,-} = \R^{n-1,1}$ identifies the time component of $x^\mu$ as
	/// $x^0$ in compliance with the convention of identifying spatial
	/// components $x^i$ with Latin alphabet indices starting from $i=1$.
	/// ```
	/// use nalgebra::{Vector4, Matrix4};
	/// use nalgebra_spacetime::LorentzianMN;
	/// use approx::assert_ulps_eq;
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
	fn metric() -> Self
	where
		ShapeConstraint: SameDimension<R, C>;

	/// Raises/Lowers *all* of its degree-1/degree-2 tensor indices.
	///
	/// Negates the appropriate components avoiding matrix multiplication.
	fn dual(&self) -> Self;

	/// Raises/Lowers its degree-1/degree-2 *row* tensor index.
	///
	/// Prefer `dual()` over `self.r_dual().c_dual()` to half negations.
	fn r_dual(&self) -> Self;

	/// Raises/Lowers its degree-1/degree-2 *column* tensor index.
	///
	/// Prefer `dual()` over `self.r_dual().c_dual()` to half negations.
	fn c_dual(&self) -> Self;

	/// Raises/Lowers *all* of its degree-1/degree-2 tensor indices *in-place*.
	///
	/// Negates the appropriate components avoiding matrix multiplication.
	fn dual_mut(&mut self);

	/// Raises/Lowers its degree-1/degree-2 *row* tensor index *in-place*.
	///
	/// Prefer `dual()` over `self.r_dual_mut().c_dual_mut()` to half negations.
	fn r_dual_mut(&mut self);

	/// Raises/Lowers its degree-1/degree-2 *column* tensor index *in-place*.
	///
	/// Prefer `dual()` over `self.r_dual_mut().c_dual_mut()` to half negations.
	fn c_dual_mut(&mut self);

	/// Lorentzian matrix multiplication of degree-1/degree-2 tensors.
	///
	/// Equals `self.c_dual() * rhs`, the metric contraction of its *column*
	/// index with `rhs`'s *row* index.
	fn contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>)
	-> MatrixMN<N, R, C2>
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: AreMultipliable<R, C, R2, C2>,
		DefaultAllocator: Allocator<N, R, C2>;

	/// Same as `contr()` but with transposed tensor indices.
	///
	/// Equals `self.r_dual().tr_mul(rhs)`, the metric contraction of its
	/// *transposed row* index with `rhs`'s *row* index.
	fn tr_contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>)
	-> MatrixMN<N, C, C2>
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
	///   * spacetime interval between two events, see `interval()`.
	fn scalar<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
	where
		R2: Dim,
		C2: Dim,
		SB: Storage<N, R2, C2>,
		ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>;

	/// Same as `scalar()` but with transposed tensor indices.
	///
	/// Equals `self.dual().tr_dot(rhs)`.
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
	fn timelike_norm(&self) -> N;

	/// Lorentzian norm of spacelike degree-1/degree-2 tensors.
	///
	/// Equals `self.scalar(self).sqrt()`.
	///
	/// If timelike, returns *NaN* or panics if `N` doesn't support it.
	fn spacelike_norm(&self) -> N;

	/// Spacetime interval between two events and region of `self`'s light cone.
	///
	/// Same as `interval_fn()` but with `N::default_epsilon()` as in:
	///
	///   * `is_present = |time| abs_diff_eq!(time, N::zero())`, and
	///   * `is_lightlike = |interval| abs_diff_eq!(interval, N::zero())`.
	fn interval(&self, rhs: &Self) -> (N, LightCone)
	where
		N: AbsDiffEq,
		ShapeConstraint: DimEq<U1, C>;

	/// Spacetime interval between two events and region of `self`'s light cone.
	///
	/// Equals `(rhs - self).scalar(&(rhs - self))` where `self` is subtracted
	/// from `rhs` to depict `rhs` in `self`'s light cone.
	///
	/// Requires you to approximate when `N` equals `N::zero()` via:
	///
	///   * `is_present` for the time component of `rhs - self`, and
	///   * `is_lightlike` for the interval.
	///
	/// Their signs are only evaluated in the `false` branches of `is_present`
	/// and `is_lightlike`.
	///
	/// See `interval()` for using defaults and [approx] for further details.
	///
	/// [approx]: https://docs.rs/approx
	fn interval_fn<P, L>(&self, rhs: &Self,
		is_present: P, is_lightlike: L)
	-> (N, LightCone)
	where
		ShapeConstraint: DimEq<U1, C>,
		P: Fn(N) -> bool,
		L: Fn(N) -> bool;

	/// $
	/// \gdef \uk {\hat u \cdot \vec K}
	/// \gdef \Lmu {\Lambda^{\mu'}\_{\phantom {\mu'} \mu}}
	/// \gdef \Lnu {(\Lambda^T)\_\nu^{\phantom \nu \nu'}}
	/// $
	/// Lorentz transformation $\Lmu(\hat u, \zeta)$ boosting degree-1/degree-2
	/// tensors to inertial `frame` of reference.
	///
	/// $$
	/// \Lmu(\hat u, \zeta) = I - \sinh \zeta (\uk) + (\cosh \zeta - 1) (\uk)^2
	/// $$
	///
	/// Where $\uk$ is the generator of the boost along $\hat{u}$ with its
	/// spatial components $(u^1, \dots, u^{n-1})$:
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
	/// x^{\mu'} = \Lmu x^\mu
	/// $$
	///
	/// Boosts degree-2 tensors by multiplying it from the left and its
	/// transpose (symmetric for pure boosts) from the right:
	///
	/// $$
	/// F^{\mu' \nu'} = \Lmu F^{\mu \nu} \Lnu
	/// $$
	fn new_boost<D>(frame: &FrameN<N, D>) -> Self
	where
		D: DimNameSub<U1>,
		ShapeConstraint: AreMultipliable<R, C, R, C> + DimEq<R, D>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>;

	/// Boosts this degree-1 tensor $x^\mu$ to inertial `frame` of reference
	/// along $\hat u$ with $\zeta$.
	///
	/// ```
	/// use nalgebra::{Vector3, Vector4};
	/// use nalgebra_spacetime::{LorentzianMN, Frame4};
	/// use approx::assert_ulps_eq;
	///
	/// let muon_rest_lifetime = Vector4::new(2.2e-6, 0.0, 0.0, 0.0);
	/// let muon_frame = Frame4::from_axis_beta(Vector3::z_axis(), 0.9952f64);
	/// let muon_lifetime = muon_rest_lifetime.boost(&muon_frame);
	/// let time_dilation_factor = muon_lifetime[0] / muon_rest_lifetime[0];
	/// assert_ulps_eq!((time_dilation_factor * 10.0).round() * 0.1, 10.2);
	/// ```
	///
	/// See `boost_mut()` for further details.
	fn boost<D>(&self, frame: &FrameN<N, D>) -> Self
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>;

	/// $
	/// \gdef \xu {(\vec x \cdot \hat u)}
	/// $
	/// Boosts this degree-1 tensor $x^\mu$ to inertial `frame` of reference
	/// along $\hat u$ with $\zeta$ *in-place*.
	///
	/// $$
	/// \begin{pmatrix}
	/// x^0 \\\\
	/// \vec x
	/// \end{pmatrix}' = \begin{pmatrix}
	/// x^0 \cosh \zeta - \xu \sinh \zeta \\\\
	/// \vec x + (\xu (\cosh \zeta - 1) - x^0 \sinh \zeta) \hat u
	/// \end{pmatrix}
	/// $$
	///
	/// Equals relativistic velocity addition $v \oplus u$ in case $x \equiv v$.
	///
	/// ```
	/// use nalgebra::Vector4;
	/// use nalgebra_spacetime::LorentzianMN;
	/// use approx::assert_ulps_eq;
	///
	/// // Arbitrary timelike four momentum.
	/// let mut momentum = Vector4::new(90.0, 5.0, 6.0, 7.0);
	///
	/// // Invariant rest mass.
	/// let mass = momentum.timelike_norm();
	/// // Trivial four momentum.
	/// let mass_at_rest = Vector4::new(mass, 0.0, 0.0, 0.0);
	///
	/// // Four velocity.
	/// let velocity = momentum / mass;
	/// // Four momentum boosted to its own frame.
	/// momentum.boost_mut(&velocity.frame());
	///
	/// // Four momentum boosted to its own frame is a trivial four momentum.
	/// assert_ulps_eq!(momentum, mass_at_rest, epsilon = 1e-15);
	/// ```
	fn boost_mut<D>(&mut self, frame: &FrameN<N, D>)
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>;

	/// With `temporal` and `spatial` components.
	fn new<D>(temporal: &N, spatial: &VectorN<N, DimNameDiff<D, U1>>)
	-> VectorN<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			StorageMut<N, D, U1, RStride = U1, CStride = D>;

	/// Velocity $u^\mu$ of inertial `frame` of reference.
	fn new_velocity<D>(frame: &FrameN<N, D>) -> VectorN<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>;

	/// Inertial frame of reference of this velocity $u^\mu$.
	fn frame(&self) -> FrameN<N, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<R, U1>>;

	/// Temporal component.
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
	fn spatial(&self)
	-> MatrixSliceMN<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>;

	/// Mutable spatial components.
	fn spatial_mut(&mut self)
	-> MatrixSliceMutMN<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		DefaultAllocator: Allocator<N, R, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			StorageMut<N, R, C, RStride = U1, CStride = R>;
}

impl<N, R, C> LorentzianMN<N, R, C> for MatrixMN<N, R, C>
where
	N: SimdRealField + Signed + Real,
	R: DimName,
	C: DimName,
	DefaultAllocator: Allocator<N, R, C>,
{
	#[inline]
	fn metric() -> Self
	where
		ShapeConstraint: SameDimension<R, C>,
	{
		let mut m = Self::identity();
		neg(unsafe { m.get_unchecked_mut((0, 0)) });
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

	fn dual_mut(&mut self) {
		if R::is::<U1>() || C::is::<U1>() {
			neg(unsafe { self.get_unchecked_mut(0) });
		} else if R::is::<C>() {
			for i in 1..R::dim() {
				neg(unsafe { self.get_unchecked_mut((i, 0)) });
				neg(unsafe { self.get_unchecked_mut((0, i)) });
			}
		} else {
			for r in 1..R::dim() {
				neg(unsafe { self.get_unchecked_mut((r, 0)) });
			}
			for c in 1..C::dim() {
				neg(unsafe { self.get_unchecked_mut((0, c)) });
			}
		}
	}

	fn r_dual_mut(&mut self) {
		for c in 0..C::dim() {
			neg(unsafe { self.get_unchecked_mut((0, c)) });
		}
	}

	fn c_dual_mut(&mut self) {
		for r in 0..R::dim() {
			neg(unsafe { self.get_unchecked_mut((r, 0)) });
		}
	}

	#[inline]
	fn contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>)
	-> MatrixMN<N, R, C2>
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
	fn tr_contr<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>)
	-> MatrixMN<N, C, C2>
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
		self.interval_fn(rhs,
			|time| abs_diff_eq!(time, N::zero()),
			|interval| abs_diff_eq!(interval, N::zero()),
		)
	}

	fn interval_fn<P, L>(&self, rhs: &Self,
		is_present: P, is_lightlike: L)
	-> (N, LightCone)
	where
		ShapeConstraint: DimEq<U1, C>,
		P: Fn(N) -> bool,
		L: Fn(N) -> bool,
	{
		let time = *unsafe { self.get_unchecked(0) };
		let difference = rhs - self;
		let interval = difference.scalar(&difference);
		let light_cone = if is_lightlike(interval) {
			if is_present(time) {
				Origin
			} else {
				if time.is_sign_positive()
					{ LightlikeFuture } else { LightlikePast }
			}
		} else {
			if interval.is_sign_positive() || is_present(time) {
				Spacelike
			} else {
				if time.is_sign_positive()
					{ TimelikeFuture } else { TimelikePast }
			}
		};
		(interval, light_cone)
	}

	fn new_boost<D>(frame: &FrameN<N, D>) -> Self
	where
		D: DimNameSub<U1>,
		ShapeConstraint: AreMultipliable<R, C, R, C> + DimEq<R, D>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let &FrameN { zeta_cosh, zeta_sinh, ref axis } = frame;
		let mut lambda = Self::zeros();
		for (i, u) in axis.iter().enumerate() {
			unsafe {
				*lambda.get_unchecked_mut((i + 1, 0)) = u.inlined_clone();
				*lambda.get_unchecked_mut((0, i + 1)) = u.inlined_clone();
			}
		}
		let uk = lambda.clone_owned();
		lambda.gemm(zeta_cosh - N::one(), &uk, &uk, -zeta_sinh);
		for i in 0..D::dim() {
			unsafe { *lambda.get_unchecked_mut((i, i)) += N::one(); }
		}
		lambda
	}

	#[inline]
	fn boost<D>(&self, frame: &FrameN<N, D>) -> Self
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let mut v = self.clone_owned();
		v.boost_mut(frame);
		v
	}

	fn boost_mut<D>(&mut self, frame: &FrameN<N, D>)
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		let &FrameN { zeta_cosh, zeta_sinh, ref axis } = frame;
		let u = axis.as_ref();
		let a = *unsafe { self.get_unchecked(0) };
		let zu = self.fixed_rows::<DimNameDiff<D, U1>>(1).dot(u);
		*unsafe { self.get_unchecked_mut(0) } = zeta_cosh * a - zeta_sinh * zu;
		let mut z = self.fixed_rows_mut::<DimNameDiff<D, U1>>(1);
		z += u * ((zeta_cosh - N::one()) * zu - zeta_sinh * a);
	}

	fn new<D>(temporal: &N, spatial: &VectorN<N, DimNameDiff<D, U1>>)
	-> VectorN<N, D>
	where
		D: DimNameSub<U1>,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>> + Allocator<N, D>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			StorageMut<N, D, U1, RStride = U1, CStride = D>,
	{
		let mut v = unsafe { VectorN::<N, D>::new_uninitialized() };
		*v.temporal_mut() = *temporal;
		v.spatial_mut().copy_from(spatial);
		v
	}

	#[inline]
	fn new_velocity<D>(frame: &FrameN<N, D>) -> VectorN<N, D>
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
	fn temporal(&self) -> &N
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
	{
		unsafe { self.get_unchecked(0) }
	}

	#[inline]
	fn temporal_mut(&mut self) -> &mut N
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
	{
		unsafe { self.get_unchecked_mut(0) }
	}

	#[inline]
	fn spatial(&self)
	-> MatrixSliceMN<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			Storage<N, R, C, RStride = U1, CStride = R>,
	{
		self.fixed_slice::<DimNameDiff<R, U1>, C>(1, 0)
	}

	#[inline]
	fn spatial_mut(&mut self)
	-> MatrixSliceMutMN<N, DimNameDiff<R, U1>, C, U1, R>
	where
		R: DimNameSub<U1>,
		ShapeConstraint: DimEq<U1, C>,
		<DefaultAllocator as Allocator<N, R, C>>::Buffer:
			StorageMut<N, R, C, RStride = U1, CStride = R>,
	{
		self.fixed_slice_mut::<DimNameDiff<R, U1>, C>(1, 0)
	}
}

#[inline]
fn neg<N: Scalar + Signed>(n: &mut N) {
	*n = -n.inlined_clone();
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

/// Inertial frame of reference in $n$-dimensional Lorentzian space
/// $\R^{-,+} = \R^{1,n-1}$.
///
/// Holds a statically sized direction axis $\hat u \in \R^{n-1}$ and two boost
/// parameters precomputed from either velocity $u^\mu$, rapidity $\vec \zeta$,
/// or velocity ratio $\vec \beta$ whether using `from_velocity()`,
/// `from_zeta()`, or `from_beta()`:
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
	axis: Unit<VectorN<N, DimNameDiff<D, U1>>>,
}

impl<N, D> FrameN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
{
	/// Inertial frame of reference with velocity $u^\mu$.
	pub fn from_velocity<R, C>(u: &MatrixMN<N, R, C>) -> Self
	where
		R: Dim,
		C: Dim,
		ShapeConstraint: SameNumberOfRows<R, D> + SameNumberOfColumns<C, U1>,
		DefaultAllocator: Allocator<N, R, C>
	{
		let mut scaled_axis = unsafe { VectorN::new_uninitialized() };
		let zeta_cosh = *unsafe { u.get_unchecked(0) };
		scaled_axis.iter_mut()
			.zip(u.fixed_rows::<DimNameDiff<D, U1>>(1).iter())
			.for_each(|(scaled_axis, &u)| *scaled_axis = u);
		let (axis, zeta_sinh) = Unit::new_and_get(scaled_axis);
		Self { zeta_cosh, zeta_sinh, axis }
	}

	/// Inertial frame of reference with rapidity $\vec \zeta$.
	#[inline]
	pub fn from_zeta(scaled_axis: VectorN<N, DimNameDiff<D, U1>>) -> Self {
		let (axis, zeta) = Unit::new_and_get(scaled_axis);
		Self::from_axis_zeta(axis, zeta)
	}

	/// Inertial frame of reference along $\hat u$ with rapidity $\zeta$.
	#[inline]
	pub fn from_axis_zeta(axis: Unit<VectorN<N, DimNameDiff<D, U1>>>, zeta: N)
	-> Self {
		Self { zeta_cosh: zeta.cosh(), zeta_sinh: zeta.sinh(), axis }
	}

	/// Inertial frame of reference with velocity ratio $\vec \beta$.
	#[inline]
	pub fn from_beta(scaled_axis: VectorN<N, DimNameDiff<D, U1>>) -> Self {
		let (axis, beta) = Unit::new_and_get(scaled_axis);
		Self::from_axis_beta(axis, beta)
	}

	/// Inertial frame of reference along $\hat u$ with velocity ratio $\beta$.
	#[inline]
	pub fn from_axis_beta(axis: Unit<VectorN<N, DimNameDiff<D, U1>>>, beta: N)
	-> Self {
		debug_assert!(-N::one() < beta && beta < N::one(),
			"Velocity ratio `beta` is out of range (-1, +1)");
		let gamma = N::one() / (N::one() - beta * beta).sqrt();
		Self { zeta_cosh: gamma, zeta_sinh: beta * gamma, axis }
	}

	/// Velocity $u^\mu$.
	pub fn velocity(&self) -> VectorN<N, D>
	where
		DefaultAllocator: Allocator<N, D>
	{
		let mut u = unsafe { VectorN::<N, D>::new_uninitialized() };
		*unsafe { u.get_unchecked_mut(0) } = self.gamma();
		u.fixed_rows_mut::<DimNameDiff<D, U1>>(1).iter_mut()
			.zip(self.axis.iter())
			.for_each(|(u, &axis)| *u = self.beta_gamma() * axis);
		u
	}

	/// Direction $\hat u$.
	#[inline]
	pub fn axis(&self) -> Unit<VectorN<N, DimNameDiff<D, U1>>> {
		self.axis.clone()
	}

	/// Rapidity $\zeta$.
	#[inline]
	pub fn zeta(&self) -> N {
		self.beta().atanh()
	}

	/// Velocity ratio $\beta$.
	#[inline]
	pub fn beta(&self) -> N {
		self.beta_gamma() / self.gamma()
	}

	/// Lorentz factor $\gamma$.
	#[inline]
	pub fn gamma(&self) -> N {
		self.zeta_cosh
	}

	/// Product of velocity ratio $\beta$ and Lorentz factor $\gamma$.
	#[inline]
	pub fn beta_gamma(&self) -> N {
		self.zeta_sinh
	}

	/// Relativistic velocity addition `self`$\oplus$`frame`.
	///
	/// Equals `self.velocity().boost(&-frame).frame()`.
	#[inline]
	pub fn compose(&self, frame: Self) -> Self
	where
		DefaultAllocator: Allocator<N, D>,
	{
		self.velocity().boost(&-frame).frame()
	}
}

impl<N> FrameN<N, U4>
where
	N: SimdRealField + Signed + Real,
	DefaultAllocator: Allocator<N, U3>,
{
	/// $
	/// \gdef \Bu {B^{\mu'}\_{\phantom {\mu'} \mu} (\vec \beta_u)}
	/// \gdef \Bv {B^{\mu''}\_{\phantom {\mu''} \mu'} (\vec \beta_v)}
	/// \gdef \Puv {u \oplus v}
	/// \gdef \Buv {B^{\mu'}\_{\phantom {\mu'} \mu} (\vec \beta_{\Puv})}
	/// \gdef \Ruv {R^{\mu''}\_{\phantom {\mu''} \mu'} (\epsilon)}
	/// \gdef \Luv {\Lambda^{\mu''}\_{\phantom {\mu''} \mu} (\vec \beta_{\Puv})}
	/// $
	/// Wigner rotation axis $\hat \beta_u \times \hat \beta_v$ and angle
	/// $\epsilon$ of the boost composition `self`$\oplus$`frame`.
	///
	/// The composition of two pure boosts, $\Bu$ to `self` followed by $\Bv$
	/// to `frame`, results in a composition of a pure boost $\Buv$ and a
	/// *non-vanishing* spatial rotation $\Ruv$ for *non-collinear* boosts:
	///
	/// $$
	/// \Luv = \Ruv \Buv = \Bv \Bu
	/// $$
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
	/// ```
	/// use nalgebra::Vector3;
	/// use nalgebra_spacetime::{LorentzianMN, Frame4};
	/// use approx::assert_relative_eq;
	///
	/// let v = Frame4::from_axis_beta(Vector3::x_axis(), 0.4f64);
	/// let u = Frame4::from_axis_beta(Vector3::y_axis(), 0.6f64);
	///
	/// let ucv = u.compose(v).axis();
	/// let vcu = v.compose(u).axis();
	///
	/// let (axis, angle) = v.axis_angle(&u);
	///
	/// assert_relative_eq!(axis, Vector3::z_axis());
	/// assert_relative_eq!(angle, ucv.angle(&vcu), epsilon = 1e-15);
	/// ```
	pub fn axis_angle(&self, frame: &Self) -> (Unit<Vector3<N>>, N) {
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

	/// Wigner rotation matrix $R(\hat \beta_u \times \hat \beta_v, \epsilon)$
	/// of the boost composition `self`$\oplus$`frame`.
	///
	/// See `axis_angle()` for further details.
	pub fn rotation(&self, frame: &Self) -> Matrix4<N>
	where
		N::Element: SimdRealField,
		DefaultAllocator: Allocator<N, U4, U4>,
	{
		let mut m = Matrix4::<N>::identity();
		let (axis, angle) = self.axis_angle(frame);
		let r = Rotation3::<N>::from_axis_angle(&axis, angle);
		m.fixed_slice_mut::<U3, U3>(1, 1).copy_from(r.matrix());
		m
	}
}

impl<N, R, C> From<MatrixMN<N, R, C>> for FrameN<N, R>
where
	N: SimdRealField + Signed + Real,
	R: DimNameSub<U1>,
	C: DimName,
	ShapeConstraint: SameNumberOfColumns<C, U1>,
	DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimNameDiff<R, U1>>,
{
	#[inline]
	fn from(u: MatrixMN<N, R, C>) -> Self {
		u.frame()
	}
}

impl<N, D> From<FrameN<N, D>> for VectorN<N, D>
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
		self.zeta_sinh = -self.zeta_sinh;
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
		self.compose(rhs)
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

/// Momentum in $n$-dimensional Lorentzian space $\R^{-,+} = \R^{1,n}$.
#[derive(Debug, PartialEq, Clone)]
pub struct MomentumN<N, D>
where
	N: Scalar,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	momentum: VectorN<N, D>,
}

impl<N, D> MomentumN<N, D>
where
	N: SimdRealField + Signed + Real,
	D: DimNameSub<U1>,
	DefaultAllocator: Allocator<N, D>,
{
	/// Momentum with `energy` as temporal and `momentum` as spatial components.
	#[inline]
	pub fn new(energy: &N, momentum: &VectorN<N, DimNameDiff<D, U1>>) -> Self
	where
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			StorageMut<N, D, U1, RStride = U1, CStride = D>,
	{
		Self { momentum: VectorN::<N, D>::new(energy, momentum) }
	}

	/// Momentum with rest `mass` at `velocity` as `velocity * mass`.
	#[inline]
	pub fn from_mass_at_velocity(mass: N, velocity: VectorN<N, D>) -> Self
	where
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		Self { momentum: velocity * mass }
	}

	/// Momentum with rest `mass` at `frame` as `frame.velocity() * mass`.
	#[inline]
	pub fn from_mass_at_frame(mass: N, frame: FrameN<N, D>) -> Self
	where
		DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
	{
		Self::from_mass_at_velocity(mass, frame.velocity())
	}

	/// Momentum with rest `mass` as temporal component.
	#[inline]
	pub fn from_mass_at_rest(mass: N) -> Self {
		let mut momentum = VectorN::<N, D>::zeros();
		*momentum.temporal_mut() = mass;
		Self { momentum }
	}

	/// Rest mass as timelike norm.
	#[inline]
	pub fn mass(&self) -> N {
		self.momentum.timelike_norm()
	}

	/// Velocity as momentum divided by rest `mass()`.
	pub fn velocity(&self) -> VectorN<N, D> {
		self.momentum.clone() / self.mass()
	}

	/// Energy as temporal component.
	#[inline]
	pub fn energy(&self) -> &N {
		self.momentum.temporal()
	}

	/// Momentum as spatial components.
	#[inline]
	pub fn momentum(&self) -> VectorSliceN<N, DimNameDiff<D, U1>, U1, D>
	where
		DefaultAllocator: Allocator<N, D, U1>,
		<DefaultAllocator as Allocator<N, D, U1>>::Buffer:
			Storage<N, D, U1, RStride = U1, CStride = D>,
	{
		self.momentum.spatial()
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
		Self { momentum: self.momentum + rhs.momentum }
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
		Self { momentum: self.momentum - rhs.momentum }
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
