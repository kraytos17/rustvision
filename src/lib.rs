pub mod mat;
pub mod activations;
pub mod arch;
pub mod cost;
pub mod nn;
pub mod batch;
pub mod region;
pub mod utils;

pub use mat::Mat;
pub use activations::Activation;
pub use arch::Arch;
pub use cost::CostFn;
pub use nn::NeuralNet;
