//! # chainrule
//!
//! A minimal automatic differentiation library in Rust,
//! inspired by the functional, composable architecture of JAX.
//!
//! ## API
//!
//! ```rust,ignore
//! use chainrule::prelude::*;
//! use ndarray::array;
//!
//! #[trace]
//! fn multiply(x: Tensor, y: Tensor) -> Tensor {
//!     x * y + 1.0
//! }
//!
//! let f = trace_fn::<f32>(multiply);
//!
//! let a = array![1., 2., 3.];
//! let b = array![4., 5., 6.];
//!
//! let out = f.eval()((a, b));
//! assert_eq!(out, &a * &b + 1.0);
//! ```
//!
//! A `dense` forward pass:
//!
//! ```rust,ignore
//! use chainrule::prelude::*;
//! use ndarray::arr2;
//!
//! #[trace]
//! fn dense(w: Tensor, x: Tensor, b: Tensor) -> Tensor {
//!     x.matmul(w) + b
//! }
//!
//! let f = trace_fn::<f32>(dense);
//!
//! let w = arr2(&[[1., 2.], [3., 4.]]);
//! let x = arr2(&[[1., 1.], [2., 2.]]);
//! let b = arr2(&[[1., 1.], [1., 1.]]);
//!
//! let out = f.eval()((&w, &x, &b));
//! let expected = x.dot(&w) + &b;
//! assert_eq!(out, expected.into_dyn());
//! ```
//!

use core::fmt::Debug;

use num_traits::{Float, NumOps};

/// Blanket floating scalar trait for tensors.
pub trait Floating: Debug + Float + NumOps {
    fn from_f64(val: f64) -> Self;
}

impl Floating for f32 {
    fn from_f64(val: f64) -> Self {
        val as f32
    }
}
impl Floating for f64 {
    fn from_f64(val: f64) -> Self {
        val
    }
}

// Internal modules
pub mod context;
pub mod graph;
pub mod identity;
pub mod ops;
pub mod tracing;

// Public API

/// Re‑export the `#[trace]` attribute macro.
pub use chainrule_macros::trace;

pub use crate::graph::Graph;
pub use crate::identity::Id;
pub use crate::tracing::function::TraceableFn;
/// Core user types: Tensor wrapper, session, function graph.
pub use crate::tracing::{Tensor, TraceSession, Tracer};

/// Build a `TraceableFn` graph from a traced function definition.
///
/// Example:
/// ```rust,ignore
/// use chainrule::prelude::*;
///
/// #[trace]
/// fn f(x: Tensor, y: Tensor) -> Tensor {
///     x + y
/// }
///
/// let t_f = trace_fn::<f32>(f);
/// ```
pub fn trace_fn<D>(builder: fn(&mut TraceSession<D>) -> (Vec<Id>, Tracer)) -> TraceableFn<D>
where
    D: Floating + 'static,
{
    let mut g = Graph::<D>::new();
    let mut sess = TraceSession::new(&mut g);

    let (inputs, output) = builder(&mut sess);
    TraceableFn {
        graph: g,
        inputs,
        outputs: vec![output.id()],
    }
}

/// A prelude that brings in the most important items.
///
/// So user code can just do:
/// ```rust
/// use chainrule::prelude::*;
/// ```
pub mod prelude {
    pub use crate::tracing::tracer::Item as _;
    pub use crate::{Tensor, trace, trace_fn};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use ndarray::{Array, Ix2, arr0, arr1, arr2, array};

    // Helper for float comparison
    fn assert_all_close(a: &ndarray::ArrayD<f32>, b: &ndarray::ArrayD<f32>, tol: f32) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "Shapes do not match.\nA: {:?}\nB: {:?}",
            a,
            b
        );
        let close = a.iter().zip(b.iter()).all(|(v1, v2)| (v1 - v2).abs() < tol);
        assert!(close, "Tensors are not close.\nA: {:?}\nB: {:?}", a, b);
    }

    #[test]
    fn test_add_op() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            (x + y).sum(vec![], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr1(&[1., 2., 3.]).into_dyn();
        let y = arr1(&[10., 20., 30.]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()((&x, &y));
        let expected = arr0((&x + &y).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x, grad_y) = traced.grad().eval()((&x, &y));
        assert_eq!(grad_x, Array::ones(x.dim()).into_dyn());
        assert_eq!(grad_y, Array::ones(y.dim()).into_dyn());
    }

    #[test]
    fn test_sub_op() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            (x - y).sum(vec![], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr1(&[5., 8.]).into_dyn();
        let y = arr1(&[2., 3.]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()((&x, &y));
        let expected = arr0((&x - &y).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x, grad_y) = traced.grad().eval()((&x, &y));
        assert_eq!(grad_x, array![1.0, 1.0].into_dyn());
        assert_eq!(grad_y, array![-1.0, -1.0].into_dyn());
    }

    #[test]
    fn test_mul_op() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            (x * y).sum(vec![], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr1(&[2., 3.]).into_dyn();
        let y = arr1(&[4., 5.]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()((&x, &y));
        let expected = arr0((&x * &y).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x, grad_y) = traced.grad().eval()((&x, &y));
        assert_eq!(grad_x, y);
        assert_eq!(grad_y, x);
    }

    #[test]
    fn test_div_op() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            (x / y).sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let x = arr1(&[10.0, 20.0]).into_dyn();
        let y = arr1(&[2.0, 5.0]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()((&x, &y));
        let expected = arr0((&x / &y).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x, grad_y) = traced.grad().eval()((&x, &y));
        assert_all_close(&grad_x, &(1.0 / &y), 1e-6);
        assert_all_close(&grad_y, &(-&x / (&y * &y)), 1e-6);
    }

    #[test]
    fn test_neg_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            (-x).sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let x = arr1(&[1.0, -2.0, 3.0]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected = arr0((-&x).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        assert_eq!(grad_x, Array::from_elem(x.dim(), -1.0).into_dyn());
    }

    #[test]
    fn test_matmul_op() {
        #[trace]
        fn f(a: Tensor, b: Tensor) -> Tensor {
            a.matmul(b).sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let a = arr2(&[[1., 2.], [3., 4.]]).into_dyn();
        let b = arr2(&[[5., 6.], [7., 8.]]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()((&a, &b));
        let a_ix2: Array<f32, Ix2> = a.clone().into_dimensionality().unwrap();
        let b_ix2: Array<f32, Ix2> = b.clone().into_dimensionality().unwrap();
        let expected = arr0(a_ix2.dot(&b_ix2).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_a, grad_b) = traced.grad().eval()((&a, &b));
        let expected_grad_a = array![[11.0, 15.0], [11.0, 15.0]].into_dyn();
        let expected_grad_b = array![[4.0, 4.0], [6.0, 6.0]].into_dyn();
        assert_all_close(&grad_a, &expected_grad_a, 1e-6);
        assert_all_close(&grad_b, &expected_grad_b, 1e-6);
    }

    #[test]
    fn test_transpose_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.t().sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let x = arr2(&[[1., 2., 3.], [4., 5., 6.]]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected = arr0(x.t().sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        assert_eq!(grad_x, Array::ones(x.dim()).into_dyn());
    }

    #[test]
    fn test_sum_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.sum(vec![1], false).sum(vec![], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr2(&[[1., 2., 3.], [4., 5., 6.]]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected = arr0(x.sum_axis(ndarray::Axis(1)).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        assert_eq!(grad_x, Array::ones(x.dim()).into_dyn());
    }

    #[test]
    fn test_log_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.log().sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let x = arr1(&[1.0, 2.0, 3.0]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected = arr0(x.mapv(f32::ln).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        assert_all_close(&grad_x, &(1.0 / &x), 1e-6);
    }

    #[test]
    fn test_exp_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.exp().sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let x = arr1(&[1.0, 2.0, 3.0]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected_fwd = x.mapv(f32::exp);
        assert_all_close(&out, &arr0(expected_fwd.sum()).into_dyn(), 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        assert_all_close(&grad_x, &expected_fwd, 1e-6);
    }

    #[test]
    fn test_relu_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.relu().sum(vec![], false)
        }
        let traced = trace_fn::<f32>(f);
        let x = arr1(&[1.0, -2.0, 0.0, 4.0]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected = arr0(x.mapv(|v| if v > 0.0 { v } else { 0.0 }).sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        assert_eq!(grad_x, array![1.0, 0.0, 0.0, 1.0].into_dyn());
    }

    #[test]
    fn test_mean_op() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.mean(vec![1], false).sum(vec![], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr2(&[[1., 3., 2.], [4., 0., 4.]]).into_dyn();

        // Forward pass
        let (out,) = traced.eval()(&x);
        let expected = arr0(x.mean_axis(ndarray::Axis(1)).unwrap().sum()).into_dyn();
        assert_all_close(&out, &expected, 1e-6);

        // Backward pass
        let (grad_x,) = traced.grad().eval()(&x);
        let n = x.shape()[1] as f32; // 3
        let expected_grad =
            array![[1.0 / n, 1.0 / n, 1.0 / n], [1.0 / n, 1.0 / n, 1.0 / n]].into_dyn();
        assert_all_close(&grad_x, &expected_grad, 1e-6);
    }

    #[test]
    fn test_higher_order_grad() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            (x * x).sum(vec![], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr1(&[3.0, 5.0]).into_dyn();

        // First derivative
        let grad_fn = traced.grad();
        let (grad1,) = grad_fn.eval()(&x);
        assert_all_close(&grad1, &(2.0 * &x), 1e-6);

        // Second derivative
        let (grad2,) = grad_fn.grad().eval()(&x);
        assert_all_close(&grad2, &array![2.0, 2.0].into_dyn(), 1e-6);
    }
}
