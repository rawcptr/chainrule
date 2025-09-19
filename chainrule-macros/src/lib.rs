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

        #[allow(unused_parens)]
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

#[derive(Debug, Clone)]
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
                let tmp_out = self.fresh("tmp_out");
                let sess = &self.sess_ident;
                match bin.op {
                    BinOp::Add(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        let #tmp_out = #sess.add(#tmp_l, #tmp_r);
                        #tmp_out
                    }},
                    BinOp::Sub(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        let #tmp_out = #sess.sub(#tmp_l, #tmp_r);
                        #tmp_out
                    }},
                    BinOp::Mul(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        let #tmp_out = #sess.mul(#tmp_l, #tmp_r);
                        #tmp_out
                    }},
                    BinOp::Div(_) => syn::parse_quote! {{
                        let #tmp_l = #lhs;
                        let #tmp_r = #rhs;
                        let #tmp_out = #sess.div(#tmp_l, #tmp_r);
                        #tmp_out
                    }},
                    _ => syn::parse_quote! {
                        compile_error!("unsupported operator in #[trace] fn")
                    },
                }
            }

            Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => {
                let inner = self.fold_expr(*u.expr);
                let tmp_in = self.fresh("tmp_in");
                let tmp_out = self.fresh("tmp_neg");
                let sess = &self.sess_ident;
                syn::parse_quote! {{
                    let #tmp_in = #inner;
                    let #tmp_out = #sess.neg(#tmp_in);
                    #tmp_out
                }}
            }

            Expr::Lit(lit) => {
                let sess = &self.sess_ident;
                if let syn::Lit::Float(lit_float) = lit.lit {
                    syn::parse_quote! {{
                        #sess.constant(D::from_f64(#lit_float))
                    }}
                } else {
                    Expr::Lit(lit)
                }
            }
            Expr::MethodCall(mc) => {
                let receiver = self.fold_expr(*mc.receiver);
                let args: Vec<_> = mc.args.into_iter().map(|a| self.fold_expr(a)).collect();
                let method = mc.method.clone();
                let recv_tmp = self.fresh("recv");
                let out_tmp = self.fresh("tmp_out");
                let arg_tmps: Vec<syn::Ident> =
                    (0..args.len()).map(|_| self.fresh("arg")).collect();
                let sess = &self.sess_ident;

                // List of ops we route through the session
                let is_traced = matches!(
                    method.to_string().as_str(),
                    "matmul"
                        | "t"
                        | "transpose"
                        | "reshape"
                        | "broadcast"
                        | "sum"
                        | "exp"
                        | "log"
                        | "relu"
                        | "div"
                        | "max"
                        | "mean"
                );

                if is_traced {
                    syn::parse_quote! {{
                        let #recv_tmp = #receiver;
                        #( let #arg_tmps = #args; )*
                        let #out_tmp = #sess.#method(#recv_tmp #(, #arg_tmps )* );
                        #out_tmp
                    }}
                } else {
                    syn::parse_quote! {{
                        let #recv_tmp = #receiver;
                        #( let #arg_tmps = #args; )*
                        let #out_tmp = #recv_tmp.#method(#( #arg_tmps ),* );
                        #out_tmp
                    }}
                }
            }
            other => fold::fold_expr(self, other),
        }
    }
}
