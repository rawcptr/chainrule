use crate::{
    Floating, Graph, Id, TraceSession, Tracer, ops::Mul, simple_unary_op, tracing::TensorData,
};

simple_unary_op!(
    Exp,
    disp: "exp",
    fwd: |x: TensorData<D>| x.mapv(|a| a.exp()),
    vjp: |this: &Exp, g: &mut Graph<D>, og: Id| {
        let out = g.fresh();
        g.push(Box::new(Exp::new(this.inp, out)));
        let prod = g.fresh();
        g.push(Box::new(Mul::new(og, out, prod)));
        prod
    }
);

impl Tracer {
    pub fn exp(&self) {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    pub fn exp(&mut self, a: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(Exp::new(a.id(), out), out)
    }
}
