use super::{AffineSimplex, SmallVector};
use crate::{errors::Error, numbers::Real};

/// Reference-cell quadrature. Weights include reference measure: one for the
/// unit interval, one half for the triangle, and one sixth for the tetrahedron.
#[derive(Clone, Debug)]
pub struct QuadratureRule<T, const D: usize> {
    points: Vec<SmallVector<T, D>>,
    weights: Vec<T>,
    degree: usize,
}

impl<T: Real, const D: usize> QuadratureRule<T, D> {
    pub fn points(&self) -> &[SmallVector<T, D>] {
        &self.points
    }
    pub fn weights(&self) -> &[T] {
        &self.weights
    }
    pub fn degree(&self) -> usize {
        self.degree
    }
    pub fn integrate<V>(&self, f: impl Fn(SmallVector<T, D>) -> V) -> V
    where
        V: num_traits::Zero + std::ops::Add<Output = V> + std::ops::Mul<T, Output = V>,
    {
        self.points
            .iter()
            .zip(&self.weights)
            .fold(V::zero(), |sum, (&point, &weight)| sum + f(point) * weight)
    }
    pub fn integrate_mapped<V>(
        &self,
        map: &AffineSimplex<T, D>,
        f: impl Fn(SmallVector<T, D>) -> V,
    ) -> V
    where
        V: num_traits::Zero + std::ops::Add<Output = V> + std::ops::Mul<T, Output = V>,
    {
        self.integrate(|point| f(map.map_point(point)) * map.jacobian().measure_scale())
    }
}

fn legendre(n: usize, z: f64) -> (f64, f64) {
    let (mut previous, mut current) = (1.0, z);
    for k in 2..=n {
        let next = ((2 * k - 1) as f64 * z * current - (k - 1) as f64 * previous) / k as f64;
        previous = current;
        current = next;
    }
    (current, n as f64 * (z * current - previous) / (z * z - 1.0))
}
fn gauss(order: usize) -> Result<Vec<(f64, f64)>, Error> {
    if !(1..=16).contains(&order) {
        return Err(Error::InvalidArgument(
            "Gauss order must be in 1..=16".into(),
        ));
    }
    let mut rule = vec![(0.0, 0.0); order];
    for i in 0..order.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..64 {
            let (p, dp) = legendre(order, z);
            let next = z - p / dp;
            if (next - z).abs() <= 4.0 * f64::EPSILON {
                z = next;
                converged = true;
                break;
            }
            z = next;
        }
        if !converged {
            return Err(Error::InvalidArgument(
                "Gauss root iteration did not converge".into(),
            ));
        }
        let (_, dp) = legendre(order, z);
        let weight = 1.0 / ((1.0 - z * z) * dp * dp);
        rule[i] = ((1.0 - z) * 0.5, weight);
        rule[order - 1 - i] = ((1.0 + z) * 0.5, weight);
    }
    Ok(rule)
}
impl<T: Real> QuadratureRule<T, 1> {
    pub fn gauss_legendre(order: usize) -> Result<Self, Error> {
        let rule = gauss(order)?;
        Ok(Self {
            points: rule
                .iter()
                .map(|&(x, _)| SmallVector([T::from_f64(x)]))
                .collect(),
            weights: rule.iter().map(|&(_, w)| T::from_f64(w)).collect(),
            degree: 2 * order - 1,
        })
    }
}
impl<T: Real> QuadratureRule<T, 2> {
    /// Symmetric rules through degree two, then a Gauss-Duffy rule. Degrees
    /// above 30 are rejected; exactness refers to total polynomial degree.
    pub fn triangle(degree: usize) -> Result<Self, Error> {
        if degree > 30 {
            return Err(Error::InvalidArgument("triangle degree exceeds 30".into()));
        }
        let (points, weights) = if degree <= 1 {
            (vec![[1.0 / 3.0; 2]], vec![0.5])
        } else if degree == 2 {
            (
                vec![
                    [1.0 / 6.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0],
                ],
                vec![1.0 / 6.0; 3],
            )
        } else {
            let rule = gauss((degree + 2).div_ceil(2))?;
            let mut points = Vec::new();
            let mut weights = Vec::new();
            for &(u, wu) in &rule {
                for &(v, wv) in &rule {
                    points.push([u, (1.0 - u) * v]);
                    weights.push(wu * wv * (1.0 - u));
                }
            }
            (points, weights)
        };
        Ok(Self {
            points: points
                .into_iter()
                .map(|p| SmallVector(p.map(T::from_f64)))
                .collect(),
            weights: weights.into_iter().map(T::from_f64).collect(),
            degree: degree.max(1),
        })
    }
}
impl<T: Real> QuadratureRule<T, 3> {
    /// Symmetric rules through degree two, then a Gauss-Duffy rule. Maximum
    /// requested degree is 29. All weights are positive.
    pub fn tetrahedron(degree: usize) -> Result<Self, Error> {
        if degree > 29 {
            return Err(Error::InvalidArgument(
                "tetrahedron degree exceeds 29".into(),
            ));
        }
        let (points, weights) = if degree <= 1 {
            (vec![[0.25; 3]], vec![1.0 / 6.0])
        } else if degree == 2 {
            let a = (5.0 + 3.0 * 5.0f64.sqrt()) / 20.0;
            let b = (5.0 - 5.0f64.sqrt()) / 20.0;
            (
                vec![[a, b, b], [b, a, b], [b, b, a], [b, b, b]],
                vec![1.0 / 24.0; 4],
            )
        } else {
            let rule = gauss((degree + 3).div_ceil(2))?;
            let mut points = Vec::new();
            let mut weights = Vec::new();
            for &(u, wu) in &rule {
                for &(v, wv) in &rule {
                    for &(w, ww) in &rule {
                        points.push([u, (1.0 - u) * v, (1.0 - u) * (1.0 - v) * w]);
                        weights.push(wu * wv * ww * (1.0 - u) * (1.0 - u) * (1.0 - v));
                    }
                }
            }
            (points, weights)
        };
        Ok(Self {
            points: points
                .into_iter()
                .map(|p| SmallVector(p.map(T::from_f64)))
                .collect(),
            weights: weights.into_iter().map(T::from_f64).collect(),
            degree: degree.max(1),
        })
    }
}
