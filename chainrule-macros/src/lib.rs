use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::{
    BinOp, Expr, ItemFn, UnOp,
    fold::{self, Fold},
    parse_macro_input,
};

fn chainrule_crate() -> proc_macro2::TokenStream {
    match crate_name("chainrule").expect("crate `chainrule` not found") {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(#ident)
        }
    }
}

#[proc_macro_attribute]
pub fn trace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;

    let old_name = format_ident!("{}_old", fn_name);
    // all inputs must be of Tensor/Tracer type.
    let arg_idents: Vec<Box<syn::Pat>> = fn_inputs
        .iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Typed(pat) => Some(pat.pat.clone()),
            _ => None,
        })
        .collect();
    let fn_body = &input_fn.block;
    let sess_ident = syn::parse_str::<syn::Ident>("sess").unwrap();

    let mut rewriter = TraceRewriter {
        sess_ident: sess_ident.clone(),
        counter: 1,
    };

    let new_body = rewriter.fold_block(*fn_body.clone());
    let chainrule = chainrule_crate();

    let expanded = quote! {
        #fn_vis fn #old_name(#fn_inputs) #fn_output {
            #( let _ = &#arg_idents; )* // touch variables and suppress unused warning
            panic!("This function is only usable through trace_fn!");
        }

        #fn_vis fn #fn_name<'a, D: #chainrule::Floating + 'static>(
            sess: &mut #chainrule::TraceSession<'a, D>,
        ) -> (Vec<#chainrule::identity::Id>, #chainrule::tracing::Tracer) {
            #( let #arg_idents = { sess.input() }; )*
            let result = { #new_body };
            (vec![#(#arg_idents.id()),*], result)
        }
    };
    TokenStream::from(expanded)
}

struct TraceRewriter {
    sess_ident: syn::Ident,
    counter: usize,
}

impl TraceRewriter {
    fn fresh(&mut self, prefix: &str) -> syn::Ident {
        let idx = self.counter;
        self.counter += 1;
        format_ident!("__{}_{}", prefix, idx)
    }
}

impl Fold for TraceRewriter {
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::Binary(bin) => {
                let lhs = self.fold_expr(*bin.left);
                let rhs = self.fold_expr(*bin.right);
                let tmp_l = self.fresh("tmp_l");
                let tmp_r = self.fresh("tmp_r");
                let sess = &self.sess_ident;
                match bin.op {
                    BinOp::Add(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        #sess.add(#tmp_l, #tmp_r)
                    }},
                    BinOp::Sub(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        #sess.sub(#lhs, #rhs)
                    }},
                    BinOp::Mul(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        #sess.mul(#lhs, #rhs)
                    }},
                    _ => syn::parse_quote! {
                        compile_error!("unsupported operator in #[trace] fn")
                    },
                }
            }
            Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => {
                let inner = self.fold_expr(*u.expr);
                let tmp = self.fresh("tmp_neg");
                let sess = &self.sess_ident;
                syn::parse_quote! {{
                    let #tmp = #inner;
                    #sess.neg(#tmp)
                }}
            }
            Expr::Lit(lit) => {
                let sess = &self.sess_ident;
                match lit.lit {
                    syn::Lit::Float(lit_float) => syn::parse_quote! {{
                        #sess.constant(D::from_f64(#lit_float))
                    }},
                    _ => syn::parse_quote! {
                        compile_error!("unsupported literal in #[trace] fn")
                    },
                }
            }
            Expr::MethodCall(mc) => {
                let receiver = self.fold_expr(*mc.receiver.clone());
                let args: Vec<_> = mc
                    .args
                    .clone()
                    .into_iter()
                    .map(|a| self.fold_expr(a))
                    .collect();
                let sess = &self.sess_ident;

                match mc.method.to_string().as_str() {
                    "matmul" => syn::parse_quote! { #sess.matmul(#receiver, #(#args),*) },
                    "t" => syn::parse_quote! { #sess.t(#receiver) },
                    "transpose" => syn::parse_quote! { #sess.transpose(#receiver, #(#args),*) },
                    "reshape" => syn::parse_quote! { #sess.reshape(#receiver, #(#args),*) },
                    "broadcast" => syn::parse_quote! { #sess.broadcast(#receiver, #(#args),*) },
                    "sum" => syn::parse_quote! { #sess.sum(#receiver, #(#args),*) },
                    "exp" => syn::parse_quote! { #sess.exp(#receiver) },
                    "log" => syn::parse_quote! { #sess.log(#receiver) },
                    "relu" => syn::parse_quote! { #sess.relu(#receiver) },
                    "div" => syn::parse_quote! { #sess.div(#receiver, #(#args),*) },
                    _ => syn::parse_quote! { #receiver.#mc.method(#(#args),*) }, // let it go untouched
                }
            }
            other => fold::fold_expr(self, other),
        }
    }
}
