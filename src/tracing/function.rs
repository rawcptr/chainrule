use crate::{Floating, context::Context, graph::Graph, identity::Id, tracing::TensorData};

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
    pub fn eval<T>(&self) -> impl Fn(T) -> TensorData<D>
    where
        T: EvalArgs<D>,
    {
        move |args: T| {
            let packed = args.pack();
            let mut ctx = Context::<D>::new();

            for (id, val) in self.inputs.iter().zip(packed.into_iter()) {
                ctx.tensors.insert(*id, val);
            }

            for op in &self.graph.nodes {
                op.eval(&mut ctx);
            }

            ctx.checked_get(&self.outputs[0]).clone()
        }
    }
}

macro_rules! as_ref_ty {
    ($_:ident, $lt:lifetime, $D:ident) => {
        &$lt TensorData<$D>
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
    // 7  => (a,b,c,d,e,f,g),
    // 8  => (a,b,c,d,e,f,g,h),
    // 9  => (a,b,c,d,e,f,g,h,i),
    // 10 => (a,b,c,d,e,f,g,h,i,j)
}
