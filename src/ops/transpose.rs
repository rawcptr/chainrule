use crate::{Floating, identity::Id, ops::Op};

#[derive(Debug, Clone)]
pub struct Transpose {
    inp: Id,
    out: Id,
    a1: usize,
    a2: usize,
}

impl Transpose {
    pub fn new(inp: Id, out: Id, a1: usize, a2: usize) -> Transpose {
        Self { inp, out, a1, a2 }
    }

    pub fn boxed(inp: Id, out: Id, a1: usize, a2: usize) -> Box<Transpose> {
        Box::new(Self::new(inp, out, a1, a2))
    }
}

impl<D: Floating> Op<D> for Transpose {
    fn name(&self) -> &str {
        "transpose"
    }

    fn eval(&self, ctx: &mut crate::context::Context<D>) {
        let mut t = ctx.checked_get(&self.inp).clone();

        t.swap_axes(self.a1, self.a2);
        ctx.tensors.insert(self.out, t);
    }

    fn vjp(&self, g: &mut crate::graph::Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let og = out_grads[0];
        let out = g.fresh();
        g.push(Self::boxed(og, out, self.a1, self.a2));
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp]
    }

    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}

#[derive(Debug, Clone)]
pub struct TransposeDefault {
    inp: Id,
    out: Id,
}

impl TransposeDefault {
    pub fn new(inp: Id, out: Id) -> TransposeDefault {
        Self { inp, out }
    }

    pub fn boxed(inp: Id, out: Id) -> Box<TransposeDefault> {
        Box::new(Self::new(inp, out))
    }
}

impl<D: Floating> Op<D> for TransposeDefault {
    fn name(&self) -> &str {
        "transpose"
    }

    fn eval(&self, ctx: &mut crate::context::Context<D>) {
        let mut t = ctx.checked_get(&self.inp).clone();
        let shape = t.shape();
        let rank = shape.len();

        if rank > 1 {
            t.swap_axes(rank - 1, rank - 1);
        };
        ctx.tensors.insert(self.out, t);
    }

    fn vjp(&self, g: &mut crate::graph::Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
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
