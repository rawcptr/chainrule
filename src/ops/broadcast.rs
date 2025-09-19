use crate::{
    Floating, Id, Tracer,
    context::Context,
    ops::{Op, sum::Sum},
    tracing::session::TraceSession,
};

#[derive(Debug, Clone)]
pub struct Broadcast {
    inp: Id,
    out: Id,
    inp_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

impl Broadcast {
    pub fn new(
        inp: Id,
        out: Id,
        target: impl Into<Vec<usize>>,
        inp_shape: impl Into<Vec<usize>>,
    ) -> Self {
        Self {
            inp,
            out,
            inp_shape: inp_shape.into(),
            target_shape: target.into(),
        }
    }
    pub fn boxed(
        inp: Id,
        out: Id,
        target: impl Into<Vec<usize>>,
        inp_shape: impl Into<Vec<usize>>,
    ) -> Box<Self> {
        Box::new(Self::new(inp, out, target, inp_shape))
    }
}

impl<D: Floating> Op<D> for Broadcast {
    fn name(&self) -> &'static str {
        "broadcast"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t = ctx.checked_get(&self.inp).clone();
        let t = t
            .broadcast(self.target_shape.clone())
            .expect("failed to broadcast. dimension mismatch")
            .to_owned();

        ctx.tensors.insert(self.out, t);
    }

    fn vjp(&self, g: &mut crate::Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let grad_y = *out_grads.first()?;
        let out = g.fresh();
        let mut axes_to_reduce = Vec::new();
        let target_rank = self.target_shape.len();
        let inp_rank = self.inp_shape.len();

        for i in 0..(target_rank - inp_rank) {
            axes_to_reduce.push(i);
        }

        for i in 0..inp_rank {
            let inp_dim = self.inp_shape.get(i)?;
            let target_dim = self.target_shape.get(i + (target_rank - inp_rank))?;
            if *inp_dim == 1 && *target_dim > 1 {
                axes_to_reduce.push(i + (target_rank - inp_rank));
            }
        }

        let sum = Box::new(Sum::new(
            grad_y,
            out,
            axes_to_reduce,
            &*self.target_shape,
            true,
        ));

        g.push(sum);
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp]
    }

    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    #[must_use]
    pub fn broadcast(&mut self, t: Tracer, shape: Vec<usize>) -> Tracer {
        let out = self.g.fresh();
        self.emit(Broadcast::new(t.id(), out, shape, t.shape()), out)
    }
}

impl Tracer {
    pub fn broadcast(&self, _: Tracer, _: Vec<usize>) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}
