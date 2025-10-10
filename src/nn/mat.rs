use rand::Rng;
use std::ops::{Add, Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);
        Self {
            rows,
            cols,
            data: vec![],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, v: Vec<f32>) -> Self {
        assert_eq!(v.len(), rows * cols);
        Self {
            rows,
            cols,
            data: v,
        }
    }

    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn row(&self, r: usize) -> &[f32] {
        let offset = r * self.cols;
        &self.data[offset..offset + self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let offset = r * self.cols;
        &mut self.data[offset..offset + self.cols]
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "Inner dims must match for matmul");
        let mut result = Self::new(self.rows, rhs.cols);

        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(r, k)];
                for c in 0..rhs.cols {
                    result[(r, c)] += a * rhs[(k, c)];
                }
            }
        }

        result
    }

    pub fn rand_matrix(rows: usize, cols: usize, min: f32, max: f32) -> Self {
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(min..max))
            .collect();

        Self::from_vec(rows, cols, data)
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        &mut self.data[r * self.cols + c]
    }
}

impl Add for &Mat {
    type Output = Mat;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Matrix shapes must match for addition"
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Mat::from_vec(self.rows, self.cols, data)
    }
}
