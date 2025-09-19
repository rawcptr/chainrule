//! # chainrule
//!
//! A minimal automatic differentiation library in Rust,
//! inspired by the functional, composable architecture of JAX.
//!
//! ## API
//!
//! ```rust,ignore
//! use chainrule::prelude;:*;
//! use ndarray::array;
//!
//! #[trace]
//! fn multiply(x: Tensor, y: Tensor) -> Tensor {
//!     x * y + 1.0
//! }
//!
//! let f = trace_fn::<f32>(multiply);
//!
//! let a = array![1., 2., 3.].into_dyn();
//! let b = array![4., 5., 6.].into_dyn();
//!
//! let out = f.eval()((&a, &b));
//! assert_eq!(out, &a * &b + 1.0);
//! ```
//!
//! A `dense` forward pass:
//!
//! ```rust,ignore
//! use chainrule::prelude::*;
//! use chainrule::trace;
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
//! let out = f.eval()((&w.into_dyn(), &x.into_dyn(), &b.into_dyn()));
//! let expected = x.dot(&w) + &b;
//! assert_eq!(out, expected.into_dyn());
//! ```

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
/// use chainrule::prelude;
///
/// #[trace]
/// fn f(x: Tensor, y: Tensor) -> Tensor {
///     x + y
/// }
///
/// let traced = trace_fn::<f32>(f);
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
    pub use crate::{Tensor, trace, trace_fn};
}
