pub trait IdGenerator {
    type Id: Copy + Eq + std::hash::Hash;
    fn fresh(&mut self) -> Self::Id;
    fn release(&mut self, id: Self::Id);
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Id(usize);

pub mod generators {
    use std::collections::VecDeque;

    use crate::identity::{self, IdGenerator};

    pub struct FreeList {
        counter: usize,
        freelist: VecDeque<usize>,
    }

    impl FreeList {
        pub fn new() -> FreeList {
            Self {
                counter: 0,
                freelist: VecDeque::new(),
            }
        }
    }

    impl Default for FreeList {
        fn default() -> Self {
            Self::new()
        }
    }

    impl IdGenerator for FreeList {
        type Id = identity::Id;

        fn fresh(&mut self) -> Self::Id {
            if let Some(id) = self.freelist.pop_front() {
                identity::Id(id)
            } else {
                self.counter += 1;
                identity::Id(self.counter)
            }
        }

        fn release(&mut self, id: Self::Id) {
            self.freelist.push_back(id.0);
        }
    }
}
