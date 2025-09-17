use crate::{binary_op, graph::Graph, identity::Id, tracing::TensorData};

binary_op!(
    Mul,
    disp: "mul",
    fwd: |x: TensorData<D>, y: TensorData<D>| x * y,
    vjp: |this: &Mul, g: &mut Graph<D>, og: Id| {
        let grad_lhs = {
            let out = g.fresh();
            g.push(Box::new(Mul::new(og, this.rhs, out)));
            out
        };
        let grad_rhs = {
            let out = g.fresh();
            g.push(Box::new(Mul::new(og, this.lhs, out)));
            out
        };
        vec![grad_lhs, grad_rhs]
    }
);

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::prelude::*;

    #[test]
    fn test_mul() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            x * y
        }

        let traced = trace_fn::<f32>(f);

        let x = arr1(&[2., 3.]).into_dyn();
        let y = arr1(&[4., 5.]).into_dyn();
        let out = traced.eval()((&x, &y));
        let expected = &x * &y;
        assert_eq!(out, expected);
    }
}
