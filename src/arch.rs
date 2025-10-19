use crate::activations::Activation;

#[derive(Debug)]
pub struct Arch {
    pub layers: Vec<usize>,
    pub activations: Vec<Activation>,
}

impl Arch {
    #[must_use]
    pub fn new(layers: Vec<usize>) -> Self {
        let layer_cnt = layers.len().saturating_sub(1);
        Self {
            layers,
            activations: vec![Activation::ReLu; layer_cnt],
        }
    }

    #[must_use]
    pub fn with_activations(mut self, activations: Vec<Activation>) -> Self {
        assert_eq!(
            activations.len(),
            self.layers.len() - 1,
            "Activation count must match number of layer connections"
        );

        self.activations = activations;
        self
    }

    #[must_use]
    pub fn input_size(&self) -> usize {
        self.layers.first().copied().unwrap_or(0)
    }

    #[must_use]
    pub fn output_size(&self) -> usize {
        self.layers.last().copied().unwrap_or(0)
    }
}
