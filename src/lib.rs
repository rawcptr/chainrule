//! API.
//!
//! ```rust
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
//!     f.eval()(&a, &b);
//!     f.grad()(&a, &b);
//!     f.grad().grad()(&a, &b);
//! }
//! ```

use num_traits::{Float, NumOps};

pub trait Floating: std::fmt::Debug + Float + NumOps {}

impl Floating for f32 {}
impl Floating for f64 {}

pub mod context;
pub mod graph;
mod identity;
pub mod ops;
pub mod tracer;
