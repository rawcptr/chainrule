use crate::{Floating, Graph, Id, TraceSession, Tracer, context::Context, ops::Op};

#[derive(Debug, Clone)]
pub struct Reshape {
    inp: Id,
    out: Id,
    inp_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

impl Reshape {
    pub fn new(
        inp: Id,
        out: Id,
        inp_shape: impl Into<Vec<usize>>,
        target_shape: impl Into<Vec<usize>>,
    ) -> Self {
        Self {
            inp,
            out,
            inp_shape: inp_shape.into(),
            target_shape: target_shape.into(),
        }
    }
}

impl<D: Floating> Op<D> for Reshape {
    fn name(&self) -> &'static str {
        "reshape"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t = ctx
            .checked_get(&self.inp)
            .to_shape(self.target_shape.clone())
            .expect("Reshape should succeed as the number of elements is preserved");

        ctx.tensors.insert(self.out, t.to_owned());
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let grad_y = *out_grads.first()?;
        let out = g.fresh();

        let reshape = Reshape::new(grad_y, out, &*self.target_shape, &*self.inp_shape);
        g.push(Box::new(reshape));

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
    pub fn reshape(&mut self, t: Tracer, shape: Vec<usize>) -> Tracer {
        let out = self.g.fresh();
        self.emit(Reshape::new(t.id(), out, t.shape(), shape), out)
    }
}

impl Tracer {
    pub fn reshape(&self, _: Tracer, _: Vec<usize>) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}
