use crate::{binary_op, graph::Graph, identity::Id, tracer::TensorData};

binary_op!(
    Add,
    "add",
    |x: TensorData<D>, y: TensorData<D>| x + y,
    |_: &Add, _g: &mut Graph<D>, og: Id| { vec![og, og] }
);
