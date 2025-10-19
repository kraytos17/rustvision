use rand::Rng;
use std::iter::FromIterator;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::{fmt, slice};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, v: Vec<f32>) -> Self {
        assert_eq!(v.len(), rows * cols);
        Self {
            rows,
            cols,
            data: v,
        }
    }

    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[must_use]
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

    #[must_use]
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    #[must_use]
    pub fn mean(&self) -> f32 {
        self.sum() / self.data.len() as f32
    }

    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[(c, r)] = self[(r, c)];
            }
        }

        result
    }

    #[must_use]
    pub fn rand_matrix(rows: usize, cols: usize, min: f32, max: f32) -> Self {
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(min..max))
            .collect();

        Self::from_vec(rows, cols, data)
    }

    #[must_use]
    pub fn xavier(rows: usize, cols: usize) -> Self {
        let limit = (6.0 / (rows as f32 + cols as f32)).sqrt();
        Self::rand_matrix(rows, cols, -limit, limit)
    }

    #[must_use]
    pub fn he(rows: usize, cols: usize) -> Self {
        let std = (2.0 / rows as f32).sqrt();
        let mut rng = rand::rng();
        let data = (0..rows * cols)
            .map(|_| rng.random_range(-std..std))
            .collect();

        Self::from_vec(rows, cols, data)
    }

    #[must_use]
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

    #[must_use]
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Self::from_vec(self.rows, self.cols, data)
    }

    pub fn map_inplace<F: Fn(f32) -> f32>(&mut self, f: F) {
        for x in &mut self.data {
            *x = f(*x);
        }
    }

    #[must_use]
    pub fn zip_map<F: Fn(f32, f32) -> f32>(&self, other: &Self, f: F) -> Self {
        assert_eq!(self.shape(), other.shape());
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| f(a, b))
            .collect();

        Self::from_vec(self.rows, self.cols, data)
    }

    #[must_use]
    pub fn mse(&self, target: &Self) -> f32 {
        assert_eq!(self.shape(), target.shape());
        self.data
            .iter()
            .zip(&target.data)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / self.data.len() as f32
    }

    pub fn iter(&self) -> slice::Iter<'_, f32> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, f32> {
        self.data.iter_mut()
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f32;
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.data[r * self.cols + c]
    }
}

macro_rules! impl_bin_op_for_mat {
    ($Trait:ident, $func:ident, $op:tt) => {
        impl $Trait<&Mat> for &Mat {
            type Output = Mat;
            fn $func(self, rhs: &Mat) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let data = self.data
                    .iter()
                    .zip(&rhs.data)
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Mat::from_vec(self.rows, self.cols, data)
            }
        }
        impl $Trait<Mat> for &Mat {
            type Output = Mat;
            fn $func(self, rhs: Mat) -> Self::Output {
                &*self $op &rhs
            }
        }
        impl $Trait<&Mat> for Mat {
            type Output = Mat;
            fn $func(self, rhs: &Mat) -> Self::Output {
                &self $op rhs
            }
        }
        impl $Trait<Mat> for Mat {
            type Output = Mat;
            fn $func(self, rhs: Mat) -> Self::Output {
                &self $op &rhs
            }
        }
    };
}

macro_rules! impl_assign_op_for_mat {
    ($Trait:ident, $func:ident, $op:tt) => {
        impl $Trait<&Mat> for Mat {
            fn $func(&mut self, rhs: &Mat) {
                assert_eq!(self.shape(), rhs.shape());
                for (a, &b) in self.data.iter_mut().zip(&rhs.data) {
                    *a $op b;
                }
            }
        }
    };
}

macro_rules! impl_scalar_ops_for_mat {
    ($Trait:ident, $func:ident, $op:tt) => {
        impl $Trait<f32> for &Mat {
            type Output = Mat;
            fn $func(self, rhs: f32) -> Self::Output {
                let data = self.data.iter().map(|&x| x $op rhs).collect();
                Mat::from_vec(self.rows, self.cols, data)
            }
        }

        impl $Trait<f32> for Mat {
            type Output = Mat;
            fn $func(self, rhs: f32) -> Self::Output {
                (&self).$func(rhs)
            }
        }
    };
}

macro_rules! impl_scalar_assign_for_mat {
    ($Trait:ident, $func:ident, $op:tt) => {
        impl $Trait<f32> for Mat {
            fn $func(&mut self, rhs: f32) {
                for x in &mut self.data {
                    *x $op rhs;
                }
            }
        }
    };
}

impl_bin_op_for_mat!(Add, add, +);
impl_bin_op_for_mat!(Sub, sub, -);
impl_assign_op_for_mat!(AddAssign, add_assign, +=);
impl_assign_op_for_mat!(SubAssign, sub_assign, -=);
impl_scalar_ops_for_mat!(Mul, mul, *);
impl_scalar_assign_for_mat!(MulAssign, mul_assign, *=);

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            writeln!(f, "{:?}", &self.row(r))?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a Mat {
    type Item = &'a f32;
    type IntoIter = slice::Iter<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Mat {
    type Item = &'a mut f32;
    type IntoIter = slice::IterMut<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl FromIterator<f32> for Mat {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let data: Vec<f32> = iter.into_iter().collect();
        let cols = data.len();
        Self::from_vec(1, cols, data)
    }
}
