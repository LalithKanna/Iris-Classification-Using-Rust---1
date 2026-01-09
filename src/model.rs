use rand::Rng;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}

fn random_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::rng();

    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| rng.random_range(-0.5..0.5))
                .collect()
        })
        .collect()
}

fn zero_vector(size: usize) -> Vec<f64> {
    vec![0.0; size]
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn mat_vec_mul(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn softmax(logits: &Vec<f64>) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exp_vals: Vec<f64> = logits
        .iter()
        .map(|x| (x - max).exp())
        .collect();

    let sum: f64 = exp_vals.iter().sum();

    exp_vals.iter().map(|x| x / sum).collect()
}

fn cross_entropy_loss(y_hat: &Vec<f64>, y: &Vec<f64>) -> f64 {
    y.iter()
        .zip(y_hat.iter())
        .map(|(t, p)| -t * p.ln())
        .sum()
}

impl NeuralNetwork {
    pub fn new() -> Self {
        let input_size = 4;
        let hidden_size = 8;
        let output_size = 3;

        NeuralNetwork {
            w1: random_matrix(hidden_size, input_size),
            b1: zero_vector(hidden_size),
            w2: random_matrix(output_size, hidden_size),
            b2: zero_vector(output_size),
        }
    }

    pub fn forward(&self, input: &[f64; 4]) -> Vec<f64> {
        let x = input.to_vec();

        // z1 = W1·x + b1
        let mut z1 = mat_vec_mul(&self.w1, &x);
        for i in 0..z1.len() {
            z1[i] += self.b1[i];
        }

        // a1 = ReLU(z1)
        let a1: Vec<f64> = z1.iter().map(|&v| relu(v)).collect();

        // z2 = W2·a1 + b2
        let mut z2 = mat_vec_mul(&self.w2, &a1);
        for i in 0..z2.len() {
            z2[i] += self.b2[i];
        }

        // softmax output
        softmax(&z2)
    }

    pub fn train_step(
        &mut self,
        x: &[f64; 4],
        y: &Vec<f64>,
        learning_rate: f64,
    ) -> f64 {
        // ---------- FORWARD PASS (store intermediates) ----------
        let x_vec = x.to_vec();

        // z1 = W1·x + b1
        let mut z1 = mat_vec_mul(&self.w1, &x_vec);
        for i in 0..z1.len() {
            z1[i] += self.b1[i];
        }

        // a1 = ReLU(z1)
        let a1: Vec<f64> = z1.iter().map(|&v| relu(v)).collect();

        // z2 = W2·a1 + b2
        let mut z2 = mat_vec_mul(&self.w2, &a1);
        for i in 0..z2.len() {
            z2[i] += self.b2[i];
        }

        // y_hat = softmax(z2)
        let y_hat = softmax(&z2);

        // ---------- LOSS ----------
        let loss = cross_entropy_loss(&y_hat, y);

        // ---------- BACKWARD PASS ----------

        // delta2 = y_hat - y
        let delta2: Vec<f64> = y_hat
            .iter()
            .zip(y.iter())
            .map(|(yh, yt)| yh - yt)
            .collect();

        // Gradients for W2 and b2
        for i in 0..self.w2.len() {
            for j in 0..self.w2[0].len() {
                self.w2[i][j] -= learning_rate * delta2[i] * a1[j];
            }
            self.b2[i] -= learning_rate * delta2[i];
        }

        // delta1 = (W2^T · delta2) * ReLU'(z1)
        let mut delta1 = vec![0.0; z1.len()];
        for i in 0..z1.len() {
            let mut sum = 0.0;
            for j in 0..delta2.len() {
                sum += self.w2[j][i] * delta2[j];
            }
            delta1[i] = sum * relu_derivative(z1[i]);
        }

        // Gradients for W1 and b1
        for i in 0..self.w1.len() {
            for j in 0..self.w1[0].len() {
                self.w1[i][j] -= learning_rate * delta1[i] * x_vec[j];
            }
            self.b1[i] -= learning_rate * delta1[i];
        }

        loss
    }

    pub fn predict(&self, x: &[f64; 4]) -> usize {
        let probs = self.forward(x);

        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }
}

pub fn save_model(
    nn: &NeuralNetwork,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(nn)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}