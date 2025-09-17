use crate::{Floating, context::Context, graph::Graph, identity::Id, ops::Op};

#[derive(Debug, Clone)]
pub struct Neg {
    inp: Id,
    out: Id,
}

impl Neg {
    pub fn new(inp: Id, out: Id) -> Neg {
        Self { inp, out }
    }
    pub fn boxed(inp: Id, out: Id) -> Box<Neg> {
        Box::new(Self::new(inp, out))
    }
}

impl<D: Floating> Op<D> for Neg {
    fn name(&self) -> &str {
        "neg"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t = ctx.checked_get(&self.inp).clone();
        ctx.tensors.insert(self.out, -t);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let og = out_grads[0];
        let out = g.fresh();
        g.push(Self::boxed(og, out));
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp]
    }

    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::prelude::*;

    #[test]
    fn test_neg() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            -x
        }

        let traced = trace_fn::<f32>(f);

        let x = arr1(&[2., -3., 4.]).into_dyn();
        let out = traced.eval()(&x);
        let expected = -&x;
        assert_eq!(out, expected);
    }
}
