use crate::{
    Floating,
    graph::Graph,
    identity::Id,
    ops::{Add, Const, Input, Mul, Neg, Op, Sub, div::Div},
    tracing::Tracer,
};

pub struct TraceSession<'graph, DType: Floating> {
    pub g: &'graph mut Graph<DType>,
}

impl<D> TraceSession<'_, D>
where
    D: Floating + 'static,
{
    pub const fn new(g: &mut Graph<D>) -> TraceSession<'_, D> {
        TraceSession { g }
    }

    #[must_use]
    pub fn emit<T: Op<D> + 'static>(&mut self, op: T, out: Id) -> Tracer {
        self.g.push(Box::new(op));
        Tracer::new(out)
    }

    #[must_use]
    pub fn input(&mut self) -> Tracer {
        let out = self.g.fresh();
        self.emit(Input::new(out), out)
    }

    #[must_use]
    pub fn constant(&mut self, val: D) -> Tracer {
        let out = self.g.fresh();
        self.emit(Const::new(val, out), out)
    }

    #[must_use]
    pub fn add(&mut self, a: Tracer, b: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Add::new(a.id(), b.id(), out), out)
    }

    #[must_use]
    pub fn sub(&mut self, a: Tracer, b: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Sub::new(a.id(), b.id(), out), out)
    }

    #[must_use]
    pub fn mul(&mut self, a: Tracer, b: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Mul::new(a.id(), b.id(), out), out)
    }

    #[must_use]
    pub fn neg(&mut self, a: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Neg::new(a.id(), out), out)
    }
    
    #[must_use]
    pub fn div(&mut self, a: Tracer, b: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Div::new(a.id(), b.id(), out), out)
    }
}
