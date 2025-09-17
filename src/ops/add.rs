use crate::{binary_op, graph::Graph, identity::Id, tracing::TensorData};

binary_op!(
    Add,
    disp: "add",
    fwd: |x: TensorData<D>, y: TensorData<D>| x + y,
    vjp: |_: &Add, _g: &mut Graph<D>, og: Id| { vec![og, og] }
);

#[cfg(test)]
mod tests {
    use chainrule_macros::trace;

    #[test]
    fn test_add() {

        #[trace]
        fn f(x: crate::Tensor, y: crate::Tensor) -> crate::Tensor {
            x + y
        }

        let traced = crate::trace_fn::<f32>(f);

        let x = ndarray::arr1(&[1., 2., 3.]).into_dyn();
        let y = ndarray::arr1(&[10., 20., 30.]).into_dyn();
        let out = traced.eval()((&x, &y));
        let expected = &x + &y;
        assert_eq!(out, expected);
    }
}
