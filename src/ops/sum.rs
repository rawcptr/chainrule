use ndarray::Axis;

use crate::{
    Floating, Graph, Id,
    context::Context,
    ops::{Op, broadcast::BroadcastLike},
};

#[derive(Debug, Clone)]
pub struct Sum {
    inp: Id,
    out: Id,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Sum {
    pub fn new(inp: Id, out: Id, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Self {
        let mut axis = axis.into();
        axis.sort_unstable_by(|a, b| b.cmp(a));
        Self {
            inp,
            out,
            axis,
            keep_dims,
        }
    }
}

impl<D: Floating> Op<D> for Sum {
    fn name(&self) -> &'static str {
        "sum"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let mut t = ctx.checked_get(&self.inp).clone();
        for axis in &self.axis {
            let a = Axis(*axis);
            t = if self.keep_dims {
                t.sum_axis(a).insert_axis(a)
            } else {
                t.sum_axis(a)
            }
        }
        ctx.tensors.insert(self.out, t);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // d/dx sum(x, axis) = broadcast_like(og, like=x)
        let grad_y = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(BroadcastLike::new(grad_y, self.inp, out)));
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp]
    }

    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}

// Reduce (sum) runtime `inp` down to the runtime shape of `like`.
#[derive(Debug, Clone)]
pub struct ReduceToLike {
    inp: Id,
    like: Id,
    out: Id,
}

impl ReduceToLike {
    pub fn new(inp: Id, like: Id, out: Id) -> Self {
        Self { inp, like, out }
    }
}

impl<D: Floating> Op<D> for ReduceToLike {
    fn name(&self) -> &'static str {
        "reduce_to_like"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        use ndarray::Axis;

        let mut t = ctx.checked_get(&self.inp).clone();
        let like = ctx.checked_get(&self.like);
        let a_shape = t.shape().to_vec();
        let b_shape = like.shape().to_vec();

        assert!(
            a_shape.len() >= b_shape.len(),
            "reduce_to_like: rank(inp) < rank(like)"
        );

        let offset = a_shape.len() - b_shape.len();
        let mut axes: Vec<usize> = (0..offset).collect();
        for i in 0..b_shape.len() {
            let a = a_shape[offset + i];
            let b = b_shape[i];
            if b == 1 && a > 1 {
                axes.push(offset + i);
            } else {
                assert!(
                    a == b,
                    "reduce_to_like: incompatible dims: inp={} like={}",
                    a,
                    b
                );
            }
        }
        axes.sort_unstable_by(|x, y| y.cmp(x));
        for ax in axes {
            t = t.sum_axis(Axis(ax));
        }
        assert_eq!(
            t.shape(),
            like.shape(),
            "reduce_to_like: shapes mismatch after reduction"
        );
        ctx.tensors.insert(self.out, t);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // grad wrt inp = broadcast_like(og, like=inp)
        let og = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(BroadcastLike::new(og, self.inp, out)));
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp, self.like]
    }
    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}
