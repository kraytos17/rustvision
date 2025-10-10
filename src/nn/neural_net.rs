use crate::nn::Mat;

#[derive(Debug, Clone)]
pub enum Activation {
    ReLu,
    Sigmoid,
    Tanh,
    None,
}

impl Activation {
    pub fn apply_inplace(&self, x: &mut Mat) {
        match self {
            Self::ReLu => x.as_slice_mut().iter_mut().for_each(|v| *v = v.max(0.0)),
            Self::Sigmoid => x
                .as_slice_mut()
                .iter_mut()
                .for_each(|v| *v = 1.0 / (1.0 + (-*v).exp())),
            Self::Tanh => x.as_slice_mut().iter_mut().for_each(|v| *v = v.tanh()),
            Self::None => {}
        }
    }
}

#[derive(Debug)]
pub struct Arch {
    pub layers: Vec<usize>,
    pub activations: Vec<Activation>,
}

impl Arch {
    pub fn new(layers: Vec<usize>) -> Self {
        let layer_cnt = layers.len().saturating_sub(1);
        Self {
            layers,
            activations: vec![Activation::ReLu; layer_cnt],
        }
    }

    pub fn with_activations(mut self, activations: Vec<Activation>) -> Self {
        assert_eq!(
            activations.len(),
            self.layers.len() - 1,
            "Activation count must match number of layer connections"
        );
        self.activations = activations;

        self
    }

    pub fn input_size(&self) -> usize {
        self.layers.first().copied().unwrap_or(0)
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().copied().unwrap_or(0)
    }
}

#[derive(Debug)]
pub struct NeuralNet {
    weights: Vec<Mat>,
    biases: Vec<Mat>,
    activations: Vec<Activation>,
}
