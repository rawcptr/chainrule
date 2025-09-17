
# chainrule

A minimal automatic differentiation library in Rust, inspired by the functional, composable architecture of JAX.

`chainrule` is an exploration of the core mechanics behind modern automatic differentiation frameworks. It is a project to understand and reconstruct these systems from first principles in a systems language.

## Project Status

Experimental:

- This project is currently in the design and early implementation phase
- The README describes the target architecture and the API as envisioned upon completion
- It is built for learning and demonstration. The API is subject to breaking changes and performance is not a primary design goal.

## Features

- Reverse-mode automatic differentiation.

- Support for higher-order gradients (`grad` of `grad`).

- Dynamic computation graph construction via function tracing.

- Operator overloading for a multi-dimensional Tensor type (powered by `ndarray`).

## API Design and Usage

The current proposed API aims to implement a **two-staged process**:

1. `#[trace]`:  a proc macro that rewrites normal Rust math into graph operations (`*`, `+`, `.sin()`, etc.).
2. `trace_fn`: takes this graph builder, captures the operations as IR once, and returns a `TraceableFn` object exposing `.eval()`, `.grad()`, etc.

```rust
use chainrule::{trace, trace_fn, Tensor};
use ndarray::array;

#[trace]
fn foo(x: Tensor, y: Tensor) -> Tensor {
    x * y + 1.0
}

fn main() {
    // 1. Convert the graph-builder function into a runnable object.
    let f = trace_fn(foo);

    // 2. Define input data.
    let a = array![1., 2., 3.];
    let b = array![4., 5., 6.];

    // 3. Evaluate the function or its gradients.
    let result = f.eval()((&a, &b));
    let df = f.grad()((&a, &b));
    let ddf = f.grad().grad()((&a, &b));


    println!("result: {:?}", result);
    println!("d/dx w.r.t inputs: {:?}", df);
    println!("d^2/dx^2 w.r.t inputs: {:?}", ddf);
}
```

Note that functions are defined using the symbolic `Tensor` type to enable tracing, but the resulting TraceableFn is executed with concrete `ndarray::Array` types

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
chainrule = { git = "https://github.com/rawcptr/chainrule.git" }
```
