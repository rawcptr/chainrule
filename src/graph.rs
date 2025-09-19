use core::fmt::{Display, Formatter, Result as FmtResult};

use crate::{
    Floating,
    identity::{Id, IdGenerator, generators::FreeList},
    ops::Op,
};

#[derive(Debug, Clone)]
pub struct Graph<DType = f32, G: IdGenerator = FreeList> {
    pub nodes: Vec<Box<dyn Op<DType>>>,
    generator: G,
}

impl<D: Floating> Graph<D> {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            generator: FreeList::new(),
        }
    }

    pub fn push(&mut self, op: Box<dyn Op<D>>) {
        self.nodes.push(op);
    }

    pub fn fresh(&mut self) -> Id {
        self.generator.fresh()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Floating> Display for Graph<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(
                f,
                "{i}: {} {:?} -> {:?}",
                node.name(),
                node.inputs(),
                node.outputs()
            )?;
        }
        Ok(())
    }
}
