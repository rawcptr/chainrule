use std::collections::HashMap;

use crate::{
    Floating, context::Context, graph::Graph, identity::Id, ops::Const, tracing::TensorData,
};

#[derive(Debug, Clone)]
pub struct TraceableFn<D: Floating> {
    pub graph: Graph<D>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
}

pub trait EvalArgs<D: Floating> {
    fn pack(self) -> Vec<TensorData<D>>;
}

impl<D: Floating + 'static> TraceableFn<D> {
    fn run<T: EvalArgs<D>, O: EvalOutputs<D>>(&self, args: T) -> O {
        let packed = args.pack();
        let mut ctx = Context::<D>::new();

        for (id, val) in self.inputs.iter().zip(packed.into_iter()) {
            ctx.tensors.insert(*id, val);
        }

        for op in &self.graph.nodes {
            op.eval(&mut ctx);
        }

        O::from_vec(
            self.outputs
                .iter()
                .map(|id| ctx.checked_get(id).clone())
                .collect(),
        )
    }
    /// plucks a single
    pub fn eval<T, O>(&self) -> impl Fn(T) -> O
    where
        T: EvalArgs<D>,
        O: EvalOutputs<D>,
    {
        move |args: T| self.run(args)
    }

    pub fn grad(&self) -> Self {
        todo!("gradients are not implemented yet.")
    }
}

pub trait EvalOutputs<D> {
    fn from_vec(f: Vec<TensorData<D>>) -> Self;
}

mod macros {
    use super::{EvalArgs, EvalOutputs, Floating, TensorData};
    macro_rules! as_owned_ty {
        ($_:ident, $D:ident) => {
            TensorData<$D>
        };
    }

    macro_rules! as_ref_ty {
        ($_:ident, $D:ident) => {
            &TensorData<$D>
        };
    }

    macro_rules! reverse_order {
    // base case
    (($last:ident), $vec:ident) => {
        let $last = $vec.pop().unwrap();
    };

    // recursive case
    (($first:ident, $($rest:ident),+), $vec:ident) => {
        reverse_order!(($($rest),+), $vec);
        let $first = $vec.pop().unwrap();
    };
    }

    macro_rules! impl_eval_outputs {
        ( $( $len:literal => ( $( $name:ident ),+ ) ),+ $(,)? ) => {
            $(
                // #[expect(unused_parens, reason = "macro complains")]
                impl<D: Floating> EvalOutputs<D>
                for ( $( as_owned_ty!($name, D) ),+ ,) {
                    fn from_vec(mut f: Vec<TensorData<D>>) -> Self {
                        assert_eq!(f.len(), $len);
                        reverse_order!(($($name),+), f);
                        ( $( $name ),+ ,)
                    }
                }
            )+
        };
    }

    macro_rules! impl_eval_args {
        ( $( $len:literal => ( $( $name:ident ),+ ) ),+ $(,)? ) => {
            $(
                #[allow(unused_parens, reason = "macro complains because impl sig has a parenthesis")]
                impl<D: Floating> EvalArgs<D>
                for ( $( as_ref_ty!($name, D) ),+ )
                {
                    fn pack(self) -> Vec<TensorData<D>> {
                        let ( $( $name ),+ ) = self;
                        vec![ $( $name.clone() ),+ ]
                    }
                }
            )+
        };
    }

    impl_eval_args! {
        1  => (a),
        2  => (a,b),
        3  => (a,b,c),
        4  => (a,b,c,d),
        5  => (a,b,c,d,e),
        6  => (a,b,c,d,e,f),
        7  => (a,b,c,d,e,f,g),
    }

    impl_eval_outputs! {
        1  => (a),
        2  => (a,b),
        3  => (a,b,c),
        4  => (a,b,c,d),
        5  => (a,b,c,d,e),
        6  => (a,b,c,d,e,f),
        7  => (a,b,c,d,e,f,g)
    }
}
