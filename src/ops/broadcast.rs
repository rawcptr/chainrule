use crate::{
    Floating, Id, Tracer,
    context::Context,
    ops::{Op, sum::ReduceToLike},
    tracing::session::TraceSession,
};

#[derive(Debug, Clone)]
pub struct Broadcast {
    inp: Id,
    out: Id,
    // inp_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

impl Broadcast {
    pub fn new(inp: Id, out: Id, target: impl Into<Vec<usize>>) -> Self {
        Self {
            inp,
            out,
            target_shape: target.into(),
        }
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

        ctx.insert(self.out, t.to_owned());
    }

    fn vjp(&self, g: &mut crate::Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // d/dx broadcast(x -> target) = reduce_to_like(og, like=x)
        let grad_y = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(ReduceToLike::new(grad_y, self.inp, out)));
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
        // self.emit(Broadcast::new(t.id(), out, shape, t.shape()), out)
        self.emit(Broadcast::new(t.id(), out, shape), out)
    }
}

impl Tracer {
    pub fn broadcast(&self, _: Tracer, _: Vec<usize>) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

// Broadcast to the runtime shape of `like`.
#[derive(Debug, Clone)]
pub struct BroadcastLike {
    inp: Id,
    like: Id,
    out: Id,
}

impl BroadcastLike {
    pub fn new(inp: Id, like: Id, out: Id) -> Self {
        Self { inp, like, out }
    }
}

impl<D: Floating> Op<D> for BroadcastLike {
    fn name(&self) -> &'static str {
        "broadcast_like"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let x = ctx.checked_get(&self.inp).clone();
        let like = ctx.checked_get(&self.like);
        let shape = like.shape().to_vec();
        let y = x
            .broadcast(shape)
            .expect("broadcast_like: incompatible shapes");
        ctx.insert(self.out, y.to_owned());
    }

    fn vjp(&self, g: &mut crate::Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // grad wrt x = reduce_to_like(og, like=x)
        let og = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(ReduceToLike::new(og, self.inp, out)));
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp, self.like]
    }
    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}
