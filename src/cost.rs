use crate::Mat;

#[derive(Clone, Debug)]
pub enum CostFn {
    MeanSquaredError,
}

impl CostFn {
    #[must_use]
    pub fn cost(&self, output: &Mat, target: &Mat) -> f32 {
        assert_eq!(
            output.shape(),
            target.shape(),
            "Output/target shape mismatch"
        );

        match self {
            Self::MeanSquaredError => {
                let n = output.as_slice().len() as f32;
                output
                    .as_slice()
                    .iter()
                    .zip(target.as_slice().iter())
                    .map(|(y_hat, y)| (y_hat - y).powi(2))
                    .sum::<f32>()
                    / n
            }
        }
    }

    #[must_use]
    pub fn grad(&self, output: &Mat, target: &Mat) -> Mat {
        assert_eq!(
            output.shape(),
            target.shape(),
            "Output/target shape mismatch"
        );

        match self {
            Self::MeanSquaredError => {
                let n = output.as_slice().len() as f32;
                let grad_data: Vec<f32> = output
                    .as_slice()
                    .iter()
                    .zip(target.as_slice().iter())
                    .map(|(y_hat, y)| 2.0 * (y_hat - y) / n)
                    .collect();

                Mat::from_vec(output.shape().0, output.shape().1, grad_data)
            }
        }
    }
}
