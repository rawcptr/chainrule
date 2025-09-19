use crate::{
    Floating,
    graph::Graph,
    identity::Id,
    ops::{
        Add, Const, Input, Mul, Neg, Op, Sub,
        matmul::MatMul,
        transpose::{Transpose, TransposeDefault},
    },
    tracing::Tracer,
};

pub struct TraceSession<'graph, DType: Floating> {
    g: &'graph mut Graph<DType>,
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

    pub fn matmul(&mut self, a: Tracer, b: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(MatMul::new(a.id(), b.id(), out), out)
    }

    pub fn t(&mut self, a: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(TransposeDefault::new(a.id(), out), out)
    }

    pub fn transpose(&mut self, a: Tracer, a1: usize, a2: usize) -> Tracer {
        let out = self.g.fresh();
        self.emit(Transpose::new(a.id(), out, a1, a2), out)
    }
}
