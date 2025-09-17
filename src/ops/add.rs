use crate::{binary_op, graph::Graph, identity::Id, tracing::TensorData};

binary_op!(
    Add,
    disp: "add",
    fwd: |x: TensorData<D>, y: TensorData<D>| x + y,
    vjp: |_: &Add, _g: &mut Graph<D>, og: Id| { vec![og, og] }
);
