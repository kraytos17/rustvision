/// Pretty-prints all weight, bias, and activation info for a NeuralNet
#[macro_export]
macro_rules! print_nn {
    ($net:expr) => {{
        println!("NeuralNet {{");
        for (i, ((w, b), act)) in $net
            .weights
            .iter()
            .zip(&$net.biases)
            .zip(&$net.activations)
            .enumerate()
        {
            let (wr, wc) = w.shape();
            let (br, bc) = b.shape();

            println!("  Layer {i} {{");
            println!("    Activation: {act:?},");
            println!("    Weights ({wr}x{wc}): [");
            for r in 0..wr {
                println!("      {:?},", &w.row(r));
            }

            println!("    ]");

            println!("    Biases ({br}x{bc}): [");
            for r in 0..br {
                println!("      {:?},", &b.row(r));
            }

            println!("    ]");
            println!("  }}");
        }

        println!("}}");
    }};
}
