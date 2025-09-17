//! API.
//!
//! ```rust,ignore
//! use chainrule::trace_fn;
//! use chainrule::Tensor
//! use ndarray::array;
//!
//! #[trace]
//! fn multiply(x: Tensor:, y: Tensor) -> Tensor {
//!     x * y + 1.0
//! }
//!
//! fn main() {
//!     let a = array![1., 2., 3.];
//!     let b = array![4., 5., 6.];
//!     let f = trace_fn(multiply);
//!     f.eval()((&a, &b));
//!     f.grad()((&a, &b));
//!     f.grad().grad()((&a, &b));
//! }
//! ```

use num_traits::{Float, NumOps};

pub trait Floating: std::fmt::Debug + Float + NumOps {
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

pub mod context;
pub mod graph;
mod identity;
pub mod ops;
pub mod tracing;
pub use chainrule_macros::trace;

use crate::function::TraceableFn;
use crate::graph::Graph;
use crate::identity::Id;
pub use crate::tracing::Tensor;
pub use crate::tracing::*;

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
