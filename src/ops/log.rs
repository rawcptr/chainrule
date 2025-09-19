use crate::{
    Floating, Graph, Id, TraceSession, Tracer,
    ops::{Const, Mul, div::Div},
    simple_unary_op,
    tracing::TensorData,
};

simple_unary_op!(
    Log,
    disp: "log",
    fwd: |x: TensorData<D>| x.mapv(|a| a.ln()),
    vjp: |this: &Log, g: &mut Graph<D>, og: Id| {
        // 1/x
        let ret = g.fresh();
        let one_id = g.fresh();
        g.push(Box::new(Const::new(D::one(), one_id)));
        let inv = g.fresh();
        g.push(Box::new(Div::new(one_id, this.inp, inv)));

        // og * (1/x)
        g.push(Box::new(Mul::new(og, inv, ret)));
        ret
    }
);

impl Tracer {
    pub fn log(&self) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    pub fn log(&mut self, a: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Log::new(a.id(), out), out)
    }
}
