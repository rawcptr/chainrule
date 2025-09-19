use crate::{Graph, identity::Id, ops::transpose::TransposeDefault};
use ndarray::{
    Array, ArrayD, ArrayView1, Ix1, Ix2, IxDyn,
    linalg::{general_mat_mul, general_mat_vec_mul},
};

use crate::{Floating, binary_op, tracing::TensorData};

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let dim_a = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
        let dim_b = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);

        if dim_a == dim_b || dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            return None;
        }
    }

    result.reverse();
    Some(result)
}

fn batched_matmul<D: Floating + 'static>(a: &ArrayD<D>, b: &ArrayD<D>) -> ArrayD<D> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    assert!(
        shape_a.len() >= 2 && shape_b.len() >= 2,
        "inputs for batched matrix mul should have rank > 2"
    );

    let (m, k1) = (shape_a[shape_a.len() - 2], shape_a[shape_a.len() - 1]);
    let (k2, n) = (shape_b[shape_b.len() - 2], shape_b[shape_b.len() - 1]);
    assert_eq!(
        k1, k2,
        "inner matrix dimensions should match for matrix mul: lhs contracted dim is {k1}, rhs is {k2}"
    );

    let batch_a = &shape_a[..shape_a.len() - 2];
    let batch_b = &shape_b[..shape_b.len() - 2];
    let batch_shape = broadcast_shapes(batch_a, batch_b)
        .expect("batch dimensions should be broadcast-compatible");

    let bc_shape_a: Vec<usize> = batch_shape.iter().copied().chain([m, k1]).collect();
    let bc_shape_b: Vec<usize> = batch_shape.iter().copied().chain([k2, n]).collect();

    let a_bc = a
        .broadcast(IxDyn(&bc_shape_a))
        .expect("broadcasting to a derived valid shape should be infallible ");
    let b_bc = b
        .broadcast(IxDyn(&bc_shape_b))
        .expect("broadcasting to a derived valid shape should be infallible ");

    let result_shape: Vec<usize> = batch_shape.iter().copied().chain([m, n]).collect();
    let mut result = ArrayD::zeros(IxDyn(&result_shape));

    let batch_elems: usize = batch_shape.iter().product();
    let a_reshaped = a_bc
        .to_shape((batch_elems, m, k1))
        .expect("reshape should succeed because the number of elements is preserved");
    let b_reshaped = b_bc
        .to_shape((batch_elems, k2, n))
        .expect("reshape should succeed because the number of elements is preserved");
    let binding = result.view_mut();
    let mut r_reshaped = binding
        .to_shape((batch_elems, m, n))
        .expect("reshape should succeed because the number of elements is preserved");

    ndarray::Zip::from(a_reshaped.outer_iter())
        .and(b_reshaped.outer_iter())
        .and(r_reshaped.outer_iter_mut())
        .for_each(|ai, bi, mut ri| {
            general_mat_mul(D::one(), &ai, &bi, D::zero(), &mut ri);
        });

    result
}

// TODO: fix all the unwraps here. I just don't want a lot of visual clutter with `.expect(..)`
pub fn matmul<D: Floating + 'static>(a: TensorData<D>, b: TensorData<D>) -> TensorData<D> {
    match (a.ndim(), b.ndim()) {
        // scalar
        (0, _) | (_, 0) => &a * &b,

        // vector dot product
        (1, 1) => {
            assert_eq!(
                a.len(),
                b.len(),
                "vectors in dot-product should have same length"
            );
            let a1: ArrayView1<D> = a
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            let b1: ArrayView1<D> = b
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            TensorData::from_elem(vec![], a1.dot(&b1))
        }

        // vector (a or b) @ matrix (a, b) -> vector (1D)
        (1, 2) => {
            let n = a.len();
            assert_eq!(
                n,
                b.shape()[0],
                "vector length should match matrix's outer dimension for vec @ mat"
            ); // (n,) @ (n,m)
            let m = b.shape()[1];
            let a1 = a
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            let b2 = b
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");

            let mut result = Array::zeros(m);
            // (1×n) × (n×m) → (m,)
            general_mat_vec_mul(D::one(), &b2.t(), &a1, D::zero(), &mut result);
            result.into_dyn()
        }

        // matrix (a, b) @ vector (a or b) -> vector (1D)
        (2, 1) => {
            let n = b.len();
            assert_eq!(
                n,
                a.shape()[1],
                "vector length should match matrix's inner dimension for mat @ vec"
            ); // (m,n) @ (n,)
            let m = a.shape()[0];
            let a2 = a
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");
            let b1 = b
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");

            let mut result = Array::zeros(m);
            general_mat_vec_mul(D::one(), &a2, &b1, D::zero(), &mut result);
            result.into_dyn()
        }

        // matrix (a,b) @ matrix (b, c) -> matrix (a, c)
        (2, 2) => {
            let (m, k1) = (a.shape()[0], a.shape()[1]);
            let (k2, n) = (b.shape()[0], b.shape()[1]);
            assert_eq!(
                k1, k2,
                "inner dimension for matrix mul should be equal but lhs({k1}) != rhs({k2})"
            );

            let a2 = a
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");
            let b2 = b
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");

            let mut result = Array::zeros((m, n));
            general_mat_mul(D::one(), &a2, &b2, D::zero(), &mut result);
            result.into_dyn()
        }

        // fallback to batched dims
        _ => batched_matmul(&a, &b),
    }
}

binary_op!(
    MatMul,
    disp: "matmul",
    fwd: |x: TensorData<D>, y: TensorData<D>| matmul(x, y),
    vjp: |this: &MatMul, g: &mut Graph<D>, og: Id| {
        // "this.lhs" and "this.rhs" are the inputs to forward op (x, y)

        // Grad w.r.t lhs: d_out @ transpose(rhs)
        let rhs_t = {
            let out = g.fresh();
            g.push(TransposeDefault::boxed(this.rhs, out));
            out
        };
        let grad_lhs = {
            let out = g.fresh();
            g.push(Box::new(MatMul::new(og, rhs_t, out)));
            out
        };

        // Grad w.r.t rhs: transpose(lhs) @ d_out
        let lhs_t = {
            let out = g.fresh();
            g.push(TransposeDefault::boxed(this.lhs, out));
            out
        };
        let grad_rhs = {
            let out = g.fresh();
            g.push(Box::new(MatMul::new(lhs_t, og, out)));
            out
        };

        vec![grad_lhs, grad_rhs]
    }
);

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use crate::prelude::*;

    #[test]
    fn test_matmul() {
        #[trace]
        fn f(x: Tensor, w: Tensor) -> Tensor {
            x.matmul(w)
        }

        let traced = trace_fn::<f32>(f);

        let x = arr2(&[[1., 2.], [3., 4.]]);
        let w = arr2(&[[5., 6.], [7., 8.]]);
        let x2 = arr2(&[[1., 2.], [3., 4.]]).into_dyn();
        let w2 = arr2(&[[5., 6.], [7., 8.]]).into_dyn();
        let (out,) = traced.eval()((&x2, &w2));
        let expected = x.dot(&w);
        assert_eq!(out, expected.into_dyn());
    }
}
