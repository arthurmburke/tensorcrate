//! Discrete Fourier transforms of host vectors: iterative radix-2
//! Cooley–Tukey for power-of-two lengths, recursive mixed-radix for other
//! composite lengths, and the direct definition for whatever is left.

use num_traits::{Float, NumCast};

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
use super::simd_dispatch;
use super::{Host, Vector};
use crate::numbers::{Coefficient, Complex};

impl<T: Coefficient> Vector<T, Host> {
    /// Discrete Fourier transform.
    ///
    /// Power-of-two lengths use iterative radix-2 Cooley–Tukey (`O(N log N)`).
    /// Other composite lengths use a recursive mixed-radix Cooley–Tukey
    /// decomposition for radices up to 15. Sub-transforms without a factor in
    /// that range use the definition directly, so every length—including zero
    /// and one—is supported.
    ///
    /// This is the conventional unnormalized forward transform:
    /// `X[k] = Σ x[n] exp(-2πikn/N)`.
    pub fn fft(&self) -> Vector<Complex<T>, Host>
    where
        T: Float,
    {
        let n = self.len();
        let mut output = self
            .as_slice()
            .iter()
            .map(|&x| Complex::new(x, <T as num_traits::Zero>::zero()))
            .collect::<Vec<_>>();

        if n <= 1 {
            return Vector::new(output);
        }

        if !n.is_power_of_two() {
            mixed_radix_fft(&mut output, false);
            return Vector::new(output);
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::radix2_fft(&mut output, -1.0) {
            return Vector::new(output);
        }

        radix2_fft(&mut output, false);
        Vector::new(output)
    }
}

impl<T: Float + Coefficient> Vector<Complex<T>, Host> {
    /// Inverse discrete Fourier transform.
    ///
    /// This uses the same radix-2 and mixed-radix Cooley–Tukey paths as
    /// [`Vector::fft`], with a direct DFT for leaves whose smallest factor
    /// exceeds 15. It is the conventional normalized inverse transform:
    /// `x[n] = (1/N) Σ X[k] exp(2πikn/N)`.
    pub fn ifft(&self) -> Vector<Complex<T>, Host> {
        let n = self.len();
        if n <= 1 {
            return self.clone();
        }

        let mut output = self.to_vec();
        if n.is_power_of_two() {
            #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
            let vectorized = simd_dispatch::radix2_fft(&mut output, 1.0);
            #[cfg(not(all(
                feature = "simd",
                any(target_arch = "aarch64", target_arch = "x86_64")
            )))]
            let vectorized = false;
            if !vectorized {
                radix2_fft(&mut output, true);
            }
        } else {
            mixed_radix_fft(&mut output, true);
        }

        let normalization = cast::<T>(n);
        for value in &mut output {
            value.real = value.real / normalization;
            value.im = value.im / normalization;
        }
        Vector::new(output)
    }
}

fn cast<T: NumCast>(value: impl NumCast) -> T {
    NumCast::from(value).expect("usize and f64 Fourier constants fit supported float types")
}

fn radix2_fft<T: Float + Coefficient>(output: &mut [Complex<T>], inverse: bool) {
    let n = output.len();

    // Bit-reversal permutation puts inputs in the order consumed by the
    // iterative butterfly stages.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            output.swap(i, j);
        }
    }

    let direction = if inverse {
        <T as num_traits::One>::one()
    } else {
        -<T as num_traits::One>::one()
    };
    let tau = cast::<T>(std::f64::consts::TAU);
    let mut len = 2;
    while len <= n {
        let angle = direction * tau / cast::<T>(len);
        let step = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut twiddle = Complex::new(
                <T as num_traits::One>::one(),
                <T as num_traits::Zero>::zero(),
            );
            for offset in 0..len / 2 {
                let even = output[start + offset];
                let odd = output[start + offset + len / 2] * twiddle;
                output[start + offset] = even + odd;
                output[start + offset + len / 2] = even - odd;
                twiddle = twiddle * step;
            }
        }
        len *= 2;
    }
}

/// Cooley–Tukey decomposition for arbitrary composite lengths. Splitting by
/// the smallest supported factor uses radices up to 15. If no supported factor
/// divides the length, the transform uses the quadratic DFT rather than a
/// high-radix stage.
fn mixed_radix_fft<T: Float + Coefficient>(data: &mut [Complex<T>], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let direction = if inverse {
        <T as num_traits::One>::one()
    } else {
        -<T as num_traits::One>::one()
    };
    let zero = Complex::new(
        <T as num_traits::Zero>::zero(),
        <T as num_traits::Zero>::zero(),
    );

    // Two allocations for the whole transform: one workspace and the root
    // table. The recursion below borrows slices of these rather than allocating
    // per node, which for `n = 1000` (2³·5³, roughly 1250 nodes) would mean
    // thousands of allocations per call.
    let unit = roots_of_unity(n, direction);
    let mut workspace = vec![zero; n];
    transform(data, &mut workspace, &unit, 1);
}

/// Transforms `data` in place, using `workspace` (same length) as scratch.
///
/// `unit` is the root table for the *top-level* length, shared by every node.
/// A node of length `m` needs the `m`-th roots, which are a stride-`stride`
/// subsequence of it: `exp(2πik/m) == unit[k · stride]` exactly when
/// `m · stride` equals the top-level length. Each descent multiplies `stride`
/// by the radix it split off, so no node ever needs a table of its own.
fn transform<T: Float + Coefficient>(
    data: &mut [Complex<T>],
    workspace: &mut [Complex<T>],
    unit: &[Complex<T>],
    stride: usize,
) {
    let m = data.len();
    if m <= 1 {
        return;
    }
    let zero = Complex::new(
        <T as num_traits::Zero>::zero(),
        <T as num_traits::Zero>::zero(),
    );

    let Some(radix) = smallest_mixed_radix(m) else {
        // No supported factor: evaluate the definition directly.
        for frequency in 0..m {
            let mut sum = zero;
            for (index, &value) in data.iter().enumerate() {
                sum = sum + value * unit[frequency * index % m * stride];
            }
            workspace[frequency] = sum;
        }
        data.copy_from_slice(&workspace[..m]);
        return;
    };

    let quotient = m / radix;

    // Gather each residue class into its own contiguous block of `workspace`.
    for residue in 0..radix {
        for index in 0..quotient {
            workspace[residue * quotient + index] = data[residue + radix * index];
        }
    }

    // Transform each block. `data` has been fully consumed by the gather, so
    // the matching block of it is free to serve as that sub-call's workspace.
    for residue in 0..radix {
        let (start, end) = (residue * quotient, (residue + 1) * quotient);
        transform(
            &mut workspace[start..end],
            &mut data[start..end],
            unit,
            stride * radix,
        );
    }

    // Combine: output frequency `low + quotient·high` sums one sample from each
    // residue class, phase-shifted by the corresponding root.
    for high in 0..radix {
        for low in 0..quotient {
            let frequency = low + quotient * high;
            let mut sum = zero;
            for residue in 0..radix {
                sum = sum
                    + workspace[residue * quotient + low] * unit[residue * frequency % m * stride];
            }
            data[frequency] = sum;
        }
    }
}

/// `exp(direction · 2πk/n)` for every `k` in `0..n`.
///
/// The transform only ever needs these `n` angles, because the exponent is
/// periodic modulo `n`. Computing them once and indexing by the reduced
/// exponent avoids a quadratic number of `cos`/`sin` calls — and is more
/// accurate besides, since the angle handed to `cos` stays inside one period
/// instead of growing to `2π·(n−1)²/n` and losing precision to argument
/// reduction.
fn roots_of_unity<T: Float + Coefficient>(n: usize, direction: T) -> Vec<Complex<T>> {
    let tau = cast::<T>(std::f64::consts::TAU);
    (0..n)
        .map(|k| {
            let angle = direction * tau * cast::<T>(k) / cast::<T>(n);
            Complex::new(angle.cos(), angle.sin())
        })
        .collect()
}

const MAX_MIXED_RADIX: usize = 15;

fn smallest_mixed_radix(n: usize) -> Option<usize> {
    // These are all primes up to MAX_MIXED_RADIX. If none divides n, its
    // smallest prime factor necessarily exceeds the mixed-radix cutoff.
    [2, 3, 5, 7, 11, 13]
        .into_iter()
        .take_while(|&factor| factor <= MAX_MIXED_RADIX)
        .find(|&factor| n.is_multiple_of(factor))
}

#[cfg(test)]
mod fft_tests {
    use super::smallest_mixed_radix;

    #[test]
    fn mixed_radix_selection_stops_above_fifteen() {
        assert_eq!(smallest_mixed_radix(2 * 17), Some(2));
        assert_eq!(smallest_mixed_radix(13 * 17), Some(13));
        assert_eq!(smallest_mixed_radix(17 * 19), None);
        assert_eq!(smallest_mixed_radix(17), None);
    }
}
