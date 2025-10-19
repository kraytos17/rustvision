use crate::Mat;

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
