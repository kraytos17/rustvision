use rustvision::{Activation, Arch, CostFn, Mat, NeuralNet, print_nn};

fn main() {
    let arch =
        Arch::new(vec![3, 5, 2]).with_activations(vec![Activation::ReLu, Activation::Sigmoid]);
    let nn = NeuralNet::from_arch(&arch);

    let input = Mat::from_vec(1, 3, vec![0.5, -0.3, 0.8]);
    println!("Input (1x3): {:?}", input.as_slice());

    let output = nn.forward(&input);
    println!("\nFinal network output: {:?}", output.as_slice());

    let target = Mat::from_vec(1, 2, vec![1.0, 0.0]);
    println!("Target: {:?}", target.as_slice());

    let cost_fn = CostFn::MeanSquaredError;
    let loss = cost_fn.cost(&output, &target);
    println!("\nLoss (MSE): {loss}");

    let grad = cost_fn.grad(&output, &target);
    println!("Gradient of loss wrt output: {:?}", grad.as_slice());

    print_nn!(nn);
}
