pub mod add;
pub mod broadcast;
pub mod constant;
pub mod input;
pub mod matmul;
pub mod mul;
pub mod neg;
pub mod reshape;
pub mod sub;
pub mod sum;
pub mod transpose;

use core::fmt::Debug;

pub use add::Add;
pub use constant::Const;
pub use input::Input;
pub use matmul::MatMul;
pub use mul::Mul;
pub use neg::Neg;
pub use reshape::Reshape;
pub use sub::Sub;
pub use sum::Sum;
pub use transpose::{Transpose, TransposeDefault};

use crate::{context::Context, graph::Graph, identity::Id};

pub trait OpClone<D> {
    fn boxed_clone(&self) -> Box<dyn Op<D>>;
}

impl<D, T> OpClone<D> for T
where
    D: num_traits::Float,
    T: 'static + Op<D> + Clone,
{
    fn boxed_clone(&self) -> Box<dyn Op<D>> {
        Box::new(self.clone())
    }
}

pub trait Op<D>: Debug + OpClone<D> {
    /// forward semantics
    fn eval(&self, ctx: &mut Context<D>);

    fn name(&self) -> &str;

    /// symbolic vector jacobian product
    /// given inputs and upstream output grads
    /// returns gradients w.r.t inputs.
    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>>;

    /// returns the input(s) to the operation.
    fn inputs(&self) -> Vec<Id>;
    /// returns the output(s) to the operation.
    fn outputs(&self) -> Vec<Id>;
}

impl<D> Clone for Box<dyn Op<D>> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

pub mod macros {
    #[macro_export]
    /// binary operation implementer
    ///
    /// fwd: |x: Tensor, y: Tensor| -> Tensor;
    /// vjp: |self, g: Graph, og: &[Id]| -> Option<Vec<ID>>;
    macro_rules! primitive_binary_op {
        ($name:ident, disp: $strname:expr, fwd: $forward:expr, vjp: $vjp_rule:expr) => {
            #[derive(Debug, Clone)]
            #[non_exhaustive]
            pub struct $name {
                pub lhs: Id,
                pub rhs: Id,
                pub out: Id,
            }

            impl $name {
                pub fn new(
                    lhs: $crate::identity::Id,
                    rhs: $crate::identity::Id,
                    out: $crate::identity::Id,
                ) -> Self {
                    Self { lhs, rhs, out }
                }
            }

            impl<D: $crate::Floating + 'static> $crate::ops::Op<D> for $name {
                fn vjp(
                    &self,
                    g: &mut $crate::graph::Graph<D>,
                    out_grads: &[$crate::identity::Id],
                ) -> Option<Vec<Id>> {
                    let og = *out_grads.first()?;
                    Some($vjp_rule(self, g, og))
                }

                fn name(&self) -> &str {
                    $strname
                }

                fn eval(&self, ctx: &mut $crate::context::Context<D>) {
                    let x = ctx.checked_get(&self.lhs).clone();
                    let y = ctx.checked_get(&self.rhs).clone();
                    ctx.tensors.insert(self.out, ($forward)(x, y));
                }

                fn inputs(&self) -> Vec<$crate::identity::Id> {
                    vec![self.lhs, self.rhs]
                }

                fn outputs(&self) -> Vec<$crate::identity::Id> {
                    vec![self.out]
                }
            }
        };
    }
}
