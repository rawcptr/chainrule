use std::collections::HashMap;

use crate::{identity::Id, tracer::TensorData};

#[derive(Debug, Clone)]
pub struct Context<D = f32> {
    pub tensors: HashMap<Id, TensorData<D>>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
