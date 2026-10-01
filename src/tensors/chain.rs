//! Multiplying a chain of matrices in its cheapest order.

use std::borrow::Cow;

use super::{Host, Matrix};
use crate::errors::Error;
use crate::numbers::Coefficient;

/// Multiply a chain of matrices in the optimal parenthesization.
///
/// The matrices are borrowed and may have different shapes, as long as each
/// one's column count is the next one's row count. The optimal order is found
/// with the classic matrix-chain dynamic program, and each product is an
/// ordinary [`Matrix::matmul`]; use [`chained_matmul_cost`] when only the
/// optimal cost is needed, since that function uses the `O(n log n)` Hu–Shing
/// solver.
///
/// Returns [`Error::Shape`] when consecutive operands do not meet, and
/// [`Error::InvalidArgument`] for a chain shorter than two matrices or one with
/// a zero extent.
pub fn chained_matmul<T: Coefficient>(
    matrices: &[&Matrix<T, Host>],
) -> Result<Matrix<T, Host>, Error> {
    let shapes: Vec<_> = matrices.iter().map(|matrix| matrix.shape()).collect();
    let dims = validate_chain(&shapes)?;

    let n = matrices.len();
    let mut costs = vec![vec![0u128; n]; n];
    let mut splits = vec![vec![0usize; n]; n];
    for span in 2..=n {
        for i in 0..=n - span {
            let j = i + span - 1;
            costs[i][j] = u128::MAX;
            for k in i..j {
                let multiplication = (dims[i] as u128)
                    .saturating_mul(dims[k + 1] as u128)
                    .saturating_mul(dims[j + 1] as u128);
                let candidate = costs[i][k]
                    .saturating_add(costs[k + 1][j])
                    .saturating_add(multiplication);
                if candidate < costs[i][j] {
                    costs[i][j] = candidate;
                    splits[i][j] = k;
                }
            }
        }
    }

    /// The product of `matrices[i..=j]`, borrowing a lone operand rather than
    /// copying it.
    fn evaluate<'m, T: Coefficient>(
        matrices: &[&'m Matrix<T, Host>],
        splits: &[Vec<usize>],
        i: usize,
        j: usize,
    ) -> Cow<'m, Matrix<T, Host>> {
        if i == j {
            return Cow::Borrowed(matrices[i]);
        }
        let k = splits[i][j];
        let left = evaluate(matrices, splits, i, k);
        let right = evaluate(matrices, splits, k + 1, j);
        Cow::Owned(left.matmul(&right))
    }

    Ok(evaluate(matrices, &splits, 0, n - 1).into_owned())
}

/// Minimum scalar-multiplication cost for a chain of matrices with these
/// `(rows, columns)` shapes, computed by the Hu–Shing `O(n log n)` optimal
/// polygon-triangulation algorithm.
///
/// Fails exactly where [`chained_matmul`] would for the same shapes.
pub fn chained_matmul_cost(shapes: &[(usize, usize)]) -> Result<u128, Error> {
    let dims = validate_chain(shapes)?;
    Ok(hu_shing::optimal_cost(&dims.iter().map(|&d| d as i128).collect::<Vec<_>>()) as u128)
}

/// The chain's boundary dimensions — the first row count, then every column
/// count — once the shapes are known to meet.
fn validate_chain(shapes: &[(usize, usize)]) -> Result<Vec<usize>, Error> {
    if shapes.len() < 2 {
        return Err(Error::InvalidArgument(
            "expected at least 2 matrices to multiply".to_string(),
        ));
    }
    let mut dims = Vec::with_capacity(shapes.len() + 1);
    dims.push(shapes[0].0);
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        if i > 0 && rows != dims[i] {
            return Err(Error::shape(format!(
                "chained_matmul dimension mismatch: matrix {} has {} columns but matrix {i} has {rows} rows",
                i - 1,
                dims[i],
            )));
        }
        dims.push(cols);
    }
    if dims.contains(&0) {
        return Err(Error::InvalidArgument(
            "matrix-chain dimensions must be positive".to_string(),
        ));
    }
    Ok(dims)
}

/// Hu–Shing's optimal weighted-polygon triangulation algorithm. A matrix
/// chain's boundary dimensions are the polygon weights.
mod hu_shing {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone, Copy)]
    struct HArc {
        u: usize,
        v: usize,
        low: usize,
        base: i128,
        mul: i128,
        num: i128,
        den: i128,
    }

    impl HArc {
        fn contains(&self, other: &HArc) -> bool {
            self.u <= other.u && other.v <= self.v
        }

        fn support(&self) -> i128 {
            self.num / self.den
        }
    }

    impl PartialEq for HArc {
        fn eq(&self, other: &Self) -> bool {
            self.support() == other.support()
        }
    }

    impl Eq for HArc {}

    impl PartialOrd for HArc {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HArc {
        fn cmp(&self, other: &Self) -> Ordering {
            self.support().cmp(&other.support())
        }
    }

    struct Solver {
        n: usize,
        w: Vec<i128>,
        cp: Vec<i128>,
        h: Vec<HArc>,
        n_arcs: usize,
        sub: Vec<usize>,
        child: Vec<Vec<usize>>,
        n_pqs: usize,
        qid: Vec<usize>,
        pq: Vec<BinaryHeap<HArc>>,
        con: Vec<Vec<HArc>>,
    }

    impl Solver {
        fn new_arc(&mut self, u: usize, v: usize) {
            debug_assert!(u <= v);
            self.n_arcs += 1;
            let low = if self.w[u] < self.w[v] { u } else { v };
            let mul = self.w[u] * self.w[v];
            let base = self.cp[v] - self.cp[u] - mul;
            self.h[self.n_arcs] = HArc {
                u,
                v,
                low,
                base,
                mul,
                num: 0,
                den: 1,
            };
        }

        fn build_tree(&mut self, arcs: &[(usize, usize)]) {
            let mut stack = Vec::new();
            self.new_arc(1, self.n + 1);
            for &(a, b) in arcs {
                self.new_arc(a, b);
                let current = self.n_arcs;
                while let Some(&top) = stack.last() {
                    if self.h[current].contains(&self.h[top]) {
                        self.child[current].push(top);
                        stack.pop();
                    } else {
                        break;
                    }
                }
                stack.push(current);
            }
            while let Some(top) = stack.pop() {
                self.child[1].push(top);
            }
        }

        fn one_sweep(&mut self) {
            let mut stack = Vec::new();
            let mut arcs = Vec::new();
            for i in 1..=self.n {
                while stack.len() >= 2 && self.w[*stack.last().unwrap()] > self.w[i] {
                    arcs.push((stack[stack.len() - 2], i));
                    stack.pop();
                }
                stack.push(i);
            }
            while stack.len() >= 4 {
                arcs.push((1, stack[stack.len() - 2]));
                stack.pop();
            }
            let arcs = arcs
                .into_iter()
                .filter(|&(a, b)| a != 1 && b != 1)
                .collect::<Vec<_>>();
            self.build_tree(&arcs);
        }

        fn prepare(&mut self) {
            let mut first = 1;
            for i in 2..=self.n {
                if self.w[i] < self.w[first] {
                    first = i;
                }
            }
            self.w[1..=self.n].rotate_left(first - 1);
            self.w[self.n + 1] = self.w[1];
            for i in 1..=self.n + 1 {
                self.cp[i] = self.w[i] * self.w[i - 1] + self.cp[i - 1];
            }
        }

        fn minimum_neighbor_product(&self, node: usize) -> i128 {
            if node == 1 {
                return self.w[1] * self.w[2] + self.w[1] * self.w[self.n];
            }
            let current = self.h[node];
            if current.u == current.low {
                match self.con[current.u].last() {
                    Some(back) if current.contains(back) => back.mul,
                    _ => self.w[current.u] * self.w[current.u + 1],
                }
            } else {
                match self.con[current.v].last() {
                    Some(back) if current.contains(back) => back.mul,
                    _ => self.w[current.v] * self.w[current.v - 1],
                }
            }
        }

        fn add_arc(&mut self, node: usize, arc: HArc) {
            let queue = self.qid[node];
            self.pq[queue].push(arc);
            self.con[arc.u].push(arc);
            self.con[arc.v].push(arc);
        }

        fn remove_arc(&mut self, node: usize) {
            let queue = self.qid[node];
            let arc = *self.pq[queue].peek().expect("remove_arc on empty queue");
            self.con[arc.u].pop();
            self.con[arc.v].pop();
            self.pq[queue].pop();
        }

        fn merge_queues(&mut self, node: usize) {
            let mut largest = usize::MAX;
            for &child in &self.child[node] {
                if largest == usize::MAX || self.sub[largest] < self.sub[child] {
                    largest = child;
                }
            }
            self.qid[node] = self.qid[largest];
            let target = self.qid[node];
            for child in self.child[node].clone() {
                if child != largest {
                    let source = std::mem::take(&mut self.pq[self.qid[child]]);
                    self.pq[target].extend(source);
                }
            }
        }

        fn solve_subtree(&mut self, node: usize) {
            self.sub[node] = 1;
            let mul = self.h[node].mul;
            let low = self.h[node].low;

            if self.child[node].is_empty() {
                self.n_pqs += 1;
                self.qid[node] = self.n_pqs;
                let den = self.h[node].base;
                let num = self.w[low] * (den + mul - self.minimum_neighbor_product(node));
                self.h[node].num = num;
                self.h[node].den = den;
                self.add_arc(node, self.h[node]);
                return;
            }

            let mut den = self.h[node].base;
            for child in self.child[node].clone() {
                self.solve_subtree(child);
                self.sub[node] += self.sub[child];
                den -= self.h[child].base;
            }
            let mut num = self.w[low] * (den + mul - self.minimum_neighbor_product(node));
            self.merge_queues(node);
            let queue = self.qid[node];

            while matches!(self.pq[queue].peek(), Some(top) if top.support() >= self.w[low]) {
                den += self.pq[queue].peek().unwrap().den;
                self.remove_arc(node);
                num = self.w[low] * (den + mul - self.minimum_neighbor_product(node));
            }
            while matches!(self.pq[queue].peek(), Some(top) if num / den <= top.support()) {
                let top = *self.pq[queue].peek().unwrap();
                den += top.den;
                self.remove_arc(node);
                num += top.num;
            }

            self.h[node].num = num;
            self.h[node].den = den;
            self.add_arc(node, self.h[node]);
        }

        fn answer(&mut self) -> i128 {
            self.solve_subtree(1);
            let queue = std::mem::take(&mut self.pq[self.qid[1]]);
            queue.into_iter().map(|arc| arc.num).sum()
        }
    }

    pub fn optimal_cost(dims: &[i128]) -> i128 {
        match dims.len() {
            0 | 1 => return 0,
            2 => return dims[0] * dims[1],
            _ => {}
        }
        let n = dims.len();
        let len = n + 3;
        let empty_arc = HArc {
            u: 0,
            v: 0,
            low: 0,
            base: 0,
            mul: 0,
            num: 0,
            den: 1,
        };
        let mut solver = Solver {
            n,
            w: vec![0; len],
            cp: vec![0; len],
            h: vec![empty_arc; len],
            n_arcs: 0,
            sub: vec![0; len],
            child: vec![Vec::new(); len],
            n_pqs: 0,
            qid: vec![0; len],
            pq: (0..len).map(|_| BinaryHeap::new()).collect(),
            con: vec![Vec::new(); len],
        };
        solver.w[1..=n].copy_from_slice(dims);
        solver.prepare();
        solver.one_sweep();
        solver.answer()
    }
}

impl<T: Coefficient> Matrix<T, Host> {
    /// Multiply a chain of matrices, whatever their shapes, in its optimal
    /// order. See [`chained_matmul`](crate::tensors::chained_matmul).
    pub fn chained_matmul(matrices: &[&Self]) -> Result<Self, Error> {
        crate::tensors::chained_matmul(matrices)
    }

    /// The optimal multiplication cost of a chain of these matrices, from
    /// their shapes alone. See
    /// [`chained_matmul_cost`](crate::tensors::chained_matmul_cost).
    pub fn chained_matmul_cost(matrices: &[&Self]) -> Result<u128, Error> {
        let shapes: Vec<_> = matrices.iter().map(|matrix| matrix.shape()).collect();
        crate::tensors::chained_matmul_cost(&shapes)
    }
}

#[cfg(test)]
mod hu_shing_tests {
    use super::hu_shing;

    fn dynamic_programming_cost(dims: &[i128]) -> i128 {
        let n = dims.len() - 1;
        let mut costs = vec![vec![0i128; n]; n];
        for span in 2..=n {
            for i in 0..=n - span {
                let j = i + span - 1;
                costs[i][j] = i128::MAX;
                for k in i..j {
                    costs[i][j] = costs[i][j]
                        .min(costs[i][k] + costs[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]);
                }
            }
        }
        costs[0][n - 1]
    }

    #[test]
    fn hu_shing_matches_the_cubic_oracle_exhaustively() {
        let choices = [1i128, 2, 3, 4];
        for matrices in 2..=6usize {
            let dimension_count = matrices + 1;
            let cases = choices.len().pow(dimension_count as u32);
            for mut code in 0..cases {
                let dims = (0..dimension_count)
                    .map(|_| {
                        let dimension = choices[code % choices.len()];
                        code /= choices.len();
                        dimension
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    hu_shing::optimal_cost(&dims),
                    dynamic_programming_cost(&dims),
                    "dims={dims:?}"
                );
            }
        }
    }

    #[test]
    fn hu_shing_matches_the_cubic_oracle_for_random_long_chains() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        for _ in 0..1_000 {
            let matrices = 2 + (random() % 30) as usize;
            let dims = (0..=matrices)
                .map(|_| 1 + (random() % 50) as i128)
                .collect::<Vec<_>>();
            assert_eq!(
                hu_shing::optimal_cost(&dims),
                dynamic_programming_cost(&dims),
                "dims={dims:?}"
            );
        }
    }
}
