use crate::{binary_op, graph::Graph, identity::Id, tracer::TensorData};

binary_op!(
    Mul,
    "Mul",
    |x: TensorData<D>, y: TensorData<D>| x * y,
    |this: &Mul, g: &mut Graph<D>, og: Id| {
        let grad_lhs = {
            let out = g.fresh();
            g.push(Box::new(Mul::new(og, this.rhs, out)));
            out
        };
        let grad_rhs = {
            let out = g.fresh();
            g.push(Box::new(Mul::new(og, this.lhs, out)));
            out
        };
        vec![grad_lhs, grad_rhs]
    }
);
