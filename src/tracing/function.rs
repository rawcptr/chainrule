use crate::{graph::Graph, identity::Id, Floating};

pub struct TraceableFn<D: Floating> {
    pub graph: Graph<D>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>
}
