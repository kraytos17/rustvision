use crate::{Activation, Arch, Mat};

#[derive(Debug)]
pub struct NeuralNet {
    pub weights: Vec<Mat>,
    pub biases: Vec<Mat>,
    pub activations: Vec<Activation>,
}

impl NeuralNet {
    #[must_use]
    pub fn from_arch(arch: &Arch) -> Self {
        let mut weights = Vec::with_capacity(arch.layers.len() - 1);
        let mut biases = Vec::with_capacity(arch.layers.len() - 1);

        for w in arch.layers.windows(2) {
            let (in_dim, out_dim) = (w[0], w[1]);
            weights.push(Mat::rand_matrix(in_dim, out_dim, -0.1, 0.1));
            biases.push(Mat::from_vec(1, out_dim, vec![0.0; out_dim]));
        }

        Self {
            weights,
            biases,
            activations: arch.activations.clone(),
        }
    }

    /// Forward pass through the *entire* network
    #[must_use]
    pub fn forward(&self, input: &Mat) -> Mat {
        let mut x = input.clone();
        for i in 0..self.weights.len() {
            x = self.forward_layer(i, &x);
        }

        x
    }

    /// Forward pass through a *single layer*
    #[must_use]
    pub fn forward_layer(&self, layer_idx: usize, input: &Mat) -> Mat {
        let w = &self.weights[layer_idx];
        let b = &self.biases[layer_idx];
        let act = &self.activations[layer_idx];

        let mut z = input.dot(w) + b;
        act.apply_inplace(&mut z);
        z
    }

    /// Forward pass through all layers, returning intermediate activations
    #[must_use]
    pub fn forward_all(&self, input: &Mat) -> Vec<Mat> {
        let mut activations = Vec::new();
        let mut x = input.clone();

        for i in 0..self.weights.len() {
            x = self.forward_layer(i, &x);
            activations.push(x.clone());
        }

        activations
    }
}
