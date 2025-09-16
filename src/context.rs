use std::collections::HashMap;

use crate::{identity::Id, tracer::TensorData};

#[derive(Debug, Clone)]
pub struct Context<D = f32> {
    pub tensors: HashMap<Id, TensorData<D>>,
}

impl<D> Context<D> {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    pub fn checked_get(&self, id: &Id) -> &TensorData<D> {
        self.tensors
            .get(id)
            .unwrap_or_else(|| panic!("tensor({id:?}) was not found in context."))
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
