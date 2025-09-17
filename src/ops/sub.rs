use crate::{binary_op, graph::Graph, identity::Id, ops::neg::Neg, tracing::TensorData};

binary_op!(
    Sub,
    disp:  "sub",
    fwd: |x: TensorData<D>, y: TensorData<D>| x - y,
    vjp: |_: &Sub, g: &mut Graph<D>, og: Id| {
        let grad_x = og;
        let grad_y = {
            let out = g.fresh();
            g.push(Neg::boxed(og, out));
            out
        };
        vec![grad_x, grad_y]
    }
);

#[cfg(test)]
mod tests {
    use chainrule_macros::trace;
    use ndarray::arr1;

    use crate::{Tensor, trace_fn};

    #[test]
    fn test_sub() {
        #[trace]
        fn f(x: Tensor, y: Tensor) -> Tensor {
            x - y
        }

        let traced = trace_fn::<f32>(f);

        let x = arr1(&[5., 6.]).into_dyn();
        let y = arr1(&[2., 3.]).into_dyn();
        let out = traced.eval()((&x, &y));
        let expected = &x - &y;
        assert_eq!(out, expected);
    }
}
