use crate::{
    graph::Graph,
    identity::Id,
    ops::{neg::Neg, sum::ReduceToLike},
    primitive_binary_op,
    tracing::TensorData,
};

primitive_binary_op!(
    Sub,
    disp:  "sub",
    fwd: |x: TensorData<D>, y: TensorData<D>| x - y,
    vjp: |this: &Sub, g: &mut Graph<D>, og: Id| {
        let grad_x = {
            let out = g.fresh();
            g.push(Box::new(ReduceToLike::new(og, this.lhs, out)));
            out
        };
        let grad_y = {
            let neg_og = g.fresh();
            g.push(Neg::boxed(og, neg_og));
            let out = g.fresh();
            g.push(Box::new(ReduceToLike::new(neg_og, this.rhs, out)));
            out
        };
        vec![grad_x, grad_y]
    }
);

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::prelude::*;

    #[test]
    fn test_sub() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            x - y
        }

        let traced = trace_fn::<f32>(f);

        let x = arr1(&[5., 6.]).into_dyn();
        let y = arr1(&[2., 3.]).into_dyn();
        let (out,) = traced.eval()((&x, &y));
        let expected = &x - &y;
        assert_eq!(out, expected);
    }
}
