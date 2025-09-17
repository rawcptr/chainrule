use crate::identity::Id;

pub type TensorData<T = f32> = ndarray::ArrayD<T>;
pub type Tensor = Tracer;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct Tracer {
    id: Id,
}

impl Tracer {
    pub fn new(id: Id) -> Tracer {
        Self { id }
    }
    pub fn id(&self) -> Id {
        self.id
    }
}
