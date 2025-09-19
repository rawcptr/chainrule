use ndarray::Axis;

use crate::{
    Floating, Graph, Id,
    context::Context,
    ops::{Op, broadcast::Broadcast},
};

#[derive(Debug, Clone)]
pub struct Sum {
    inp: Id,
    out: Id,
    inp_shape: Vec<usize>,
    out_shape: Vec<usize>,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Sum {
    pub fn new(
        inp: Id,
        out: Id,
        axis: impl Into<Vec<usize>>,
        inp_shape: impl Into<Vec<usize>>,
        keep_dims: bool,
    ) -> Self {
        let mut axis = axis.into();
        axis.sort_unstable_by(|a, b| b.cmp(a));
        let inp_shape = inp_shape.into();
        let out_shape = inp_shape.iter().enumerate();
        let out_shape: Vec<usize> = if keep_dims {
            out_shape
                .map(|(i, &dim)| if axis.contains(&i) { 1 } else { dim })
                .collect()
        } else {
            out_shape
                .filter_map(|(i, &dim)| (!axis.contains(&i)).then_some(dim))
                .collect()
        };

        Self {
            inp,
            out,
            axis,
            inp_shape,
            out_shape,
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
        let grad_y = *out_grads.first()?;
        let out = g.fresh();
        let broadcast = Broadcast::new(grad_y, out, &*self.inp_shape, &*self.out_shape);
        g.push(Box::new(broadcast));
        Some(vec![out])
    }

    fn inputs(&self) -> Vec<Id> {
        vec![self.inp]
    }

    fn outputs(&self) -> Vec<Id> {
        vec![self.out]
    }
}
