use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, parse_macro_input};

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

    let expanded = quote! {
        #fn_vis fn #old_name(#fn_inputs) #fn_output {
            #( let _ = &#arg_idents; )* // touch variables and suppress unused warning
            panic!("This function is only usable through trace_fn!");
        }

        #fn_vis fn #fn_name<'a, D: crate::Floating + 'static>(
            sess: &mut crate::TraceSession<'a, D>,
        ) -> (Vec<crate::identity::Id>, crate::tracer::Tracer) {
            #( let #arg_idents = { sess.input() }; )*
            let result = { #fn_body };
            (vec![#(#arg_idents.id()),*], result)
        }
    };
    TokenStream::from(expanded)
}
