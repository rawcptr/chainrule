use crate::{binary_op, graph::Graph, identity::Id, ops::neg::Neg, tracing::TensorData};

binary_op!(
    Sub,
    disp:  "sub",
    fwd: |x: TensorData<D>, y: TensorData<D>| x - y,
    vjp: |_: &Sub, g: &mut Graph<D>, og: Id| {
        let grad_x = og;
        let grad_y = {
            let out = g.fresh();
            g.push(Neg::boxed(og, out));
            out
        };
        vec![grad_x, grad_y]
    }
);
