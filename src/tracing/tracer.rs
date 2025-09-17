use crate::identity::Id;

use std::ops::{Add, Mul, Neg, Sub};

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

impl Tracer {
    pub fn t(&self) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }

    pub fn transpose(&self, _a1: usize, _a2: usize) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }

    pub fn matmul(&self, _other: &Tracer) {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl Add for Tracer {
    type Output = Tracer;
    fn add(self, _rhs: Tracer) -> Tracer {
        panic!("dummy Add operator – only allowed inside #[trace] functions")
    }
}

impl Sub for Tracer {
    type Output = Tracer;
    fn sub(self, _rhs: Tracer) -> Tracer {
        panic!("dummy Sub operator – only allowed inside #[trace] functions")
    }
}

impl Mul for Tracer {
    type Output = Tracer;
    fn mul(self, _rhs: Tracer) -> Tracer {
        panic!("dummy Mul operator – only allowed inside #[trace] functions")
    }
}

impl Neg for Tracer {
    type Output = Tracer;
    fn neg(self) -> Tracer {
        panic!("dummy Neg operator – only allowed inside #[trace] functions")
    }
}
