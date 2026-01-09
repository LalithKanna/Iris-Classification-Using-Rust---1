use csv::ReaderBuilder;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::rng;
use rand::Rng;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct Sample {
    pub features: [f64; 4],
    pub label: usize,
}

#[derive(Debug)]
pub struct TrainingSample {
    pub x: [f64; 4],   
    pub y: Vec<f64>,  
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}


fn map_label(label: &str) -> usize {
    match label {
        "Iris-setosa" => 0,
        "Iris-versicolor" => 1,
        "Iris-virginica" => 2,
        _ => panic!("Unknown class label: {}", label),
    }
}

pub fn load_iris(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true) 
        .from_path(path)?;

    let mut dataset = Vec::new();

    for record in reader.records() {
        let r = record?;
   
        if r.len() != 6 {
            return Err(format!("Invalid record length: {}", r.len()).into());
        }

        let features = [
            r[1].parse::<f64>()?, 
            r[2].parse::<f64>()?, 
            r[3].parse::<f64>()?, 
            r[4].parse::<f64>()?, 
        ];

        let label = map_label(&r[5]); 
        dataset.push(Sample { features, label });
    }

    Ok(dataset)
}

pub fn shuffle_and_split(
    mut dataset: Vec<Sample>,
    train_ratio: f64,
) -> (Vec<Sample>, Vec<Sample>) {
    let mut rng = rng();
    dataset.shuffle(&mut rng);
    let train_size = (dataset.len() as f64 * train_ratio).round() as usize;

    let train_set = dataset[..train_size].to_vec();
    let test_set = dataset[train_size..].to_vec();

    (train_set, test_set)
}

pub fn compute_mean_std(train_set: &Vec<Sample>) -> ([f64; 4], [f64; 4]) {
    let n = train_set.len() as f64;

    let mut mean = [0.0; 4];
    let mut std = [0.0; 4];
    for sample in train_set {
        for i in 0..4 {
            mean[i] += sample.features[i];
        }
    }

    for i in 0..4 {
        mean[i] /= n;
    }

    for sample in train_set {
        for i in 0..4 {
            let diff = sample.features[i] - mean[i];
            std[i] += diff * diff;
        }
    }


    for i in 0..4 {
        std[i] = (std[i] / n).sqrt();

        if std[i] == 0.0 {
            std[i] = 1.0;
        }
    }

    (mean, std)
}


pub fn normalize_features(
    features: [f64; 4],
    mean: &[f64; 4],
    std: &[f64; 4],
) -> [f64; 4] {
    let mut normalized = [0.0; 4];

    for i in 0..4 {
        normalized[i] = (features[i] - mean[i]) / std[i];
    }

    normalized
}

pub fn normalize_dataset(
    dataset: &Vec<Sample>,
    mean: &[f64; 4],
    std: &[f64; 4],
) -> Vec<Sample> {
    dataset
        .iter()
        .map(|sample| Sample {
            features: normalize_features(sample.features, mean, std),
            label: sample.label,
        })
        .collect()
}

pub fn one_hot(label: usize, num_classes: usize) -> Vec<f64> {
    let mut encoded = vec![0.0; num_classes];
    encoded[label] = 1.0;
    encoded
}

pub fn to_training_samples(
    dataset: &Vec<Sample>,
    num_classes: usize,
) -> Vec<TrainingSample> {
    dataset
        .iter()
        .map(|sample| TrainingSample {
            x: sample.features,
            y: one_hot(sample.label, num_classes),
        })
        .collect()
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

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
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

pub fn accuracy(
    nn: &NeuralNetwork,
    dataset: &Vec<TrainingSample>,
) -> f64 {
    let mut correct = 0;

    for sample in dataset {
        let prediction = nn.predict(&sample.x);

        let true_label = sample
            .y
            .iter()
            .position(|&v| v == 1.0)
            .unwrap();

        if prediction == true_label {
            correct += 1;
        }
    }

    correct as f64 / dataset.len() as f64
}

fn mean(values: &Vec<f64>) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &Vec<f64>, mean: f64) -> f64 {
    let variance = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

struct ExperimentResult {
    train_acc: f64,
    test_acc: f64,
    losses: Vec<f64>,
    train_accs: Vec<f64>,
    test_accs: Vec<f64>,
    confusion_matrix: Vec<Vec<usize>>,
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

fn run_experiment(dataset: &Vec<Sample>) -> ExperimentResult {
    let (train_set, test_set) = shuffle_and_split(dataset.clone(), 0.8);

    let (mean_feat, std_feat) = compute_mean_std(&train_set);

    let train_norm = normalize_dataset(&train_set, &mean_feat, &std_feat);
    let test_norm = normalize_dataset(&test_set, &mean_feat, &std_feat);

    let train_nn = to_training_samples(&train_norm, 3);
    let test_nn = to_training_samples(&test_norm, 3);

    let mut nn = NeuralNetwork::new();
    let lr = 0.01;
    let epochs = 1000;

    let mut losses = Vec::new();
    let mut train_accs = Vec::new();
    let mut test_accs = Vec::new();

    for _ in 0..epochs {
        let mut epoch_loss = 0.0;
        for sample in &train_nn {
            epoch_loss += nn.train_step(&sample.x, &sample.y, lr);
        }
        losses.push(epoch_loss);
        train_accs.push(accuracy(&nn, &train_nn));
        test_accs.push(accuracy(&nn, &test_nn));
    }
    save_model(&nn, "trained_model.json").unwrap();
    let train_acc = accuracy(&nn, &train_nn);
    let test_acc = accuracy(&nn, &test_nn);
    let cm = confusion_matrix(&nn, &test_nn, 3);

    ExperimentResult {
        train_acc,
        test_acc,
        losses,
        train_accs,
        test_accs,
        confusion_matrix: cm,
    }
}

fn confusion_matrix(
    nn: &NeuralNetwork,
    dataset: &Vec<TrainingSample>,
    num_classes: usize,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; num_classes]; num_classes];

    for sample in dataset {
        let pred = nn.predict(&sample.x);
        let true_label = sample
            .y
            .iter()
            .position(|&v| v == 1.0)
            .unwrap();

        matrix[true_label][pred] += 1;
    }

    matrix
}

fn save_confusion_matrix(matrix: &Vec<Vec<usize>>, path: &str) {
    let mut file = File::create(path).unwrap();

    for row in matrix {
        let line = row
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{}", line).unwrap();
    }
}

fn save_curve(values: &Vec<f64>, path: &str) {
    let mut file = File::create(path).unwrap();
    for (epoch, val) in values.iter().enumerate() {
        writeln!(file, "{},{}", epoch, val).unwrap();
    }
}

fn save_values(values: &Vec<f64>, path: &str) {
    let mut file = File::create(path).unwrap();
    for v in values {
        writeln!(file, "{}", v).unwrap();
    }
}

fn save_summary(
    train_mean: f64,
    train_std: f64,
    test_mean: f64,
    test_std: f64,
) {
    let mut file = File::create("summary.csv").unwrap();
    writeln!(file, "Metric,Mean (%),Std (%)").unwrap();
    writeln!(file, "Train,{:.2},{:.2}", train_mean*100.0, train_std*100.0).unwrap();
    writeln!(file, "Test,{:.2},{:.2}", test_mean*100.0, test_std*100.0).unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = load_iris("data/iris.csv")?;
    assert_eq!(dataset.len(), 150);

    let runs = 30;

    let mut train_accuracies = Vec::new();
    let mut test_accuracies = Vec::new();

    // Run ONE experiment for plots
    let first_run = run_experiment(&dataset);

    save_confusion_matrix(&first_run.confusion_matrix, "confusion_matrix.csv");
    save_curve(&first_run.losses, "loss_curve.csv");
    save_curve(&first_run.train_accs, "train_accuracy.csv");
    save_curve(&first_run.test_accs, "test_accuracy.csv");

    // Multiple runs for statistics
    for i in 0..runs {
        let result = run_experiment(&dataset);

        train_accuracies.push(result.train_acc);
        test_accuracies.push(result.test_acc);

        println!(
            "Run {:2}: Train = {:.2}% | Test = {:.2}%",
            i + 1,
            result.train_acc * 100.0,
            result.test_acc * 100.0
        );
    }

    save_values(&test_accuracies, "test_accuracy_distribution.csv");

    let train_mean = mean(&train_accuracies);
    let train_std = std(&train_accuracies, train_mean);
    let test_mean = mean(&test_accuracies);
    let test_std = std(&test_accuracies, test_mean);

    save_summary(train_mean, train_std, test_mean, test_std);

    println!(
        "\nTrain Accuracy = {:.2}% ± {:.2}%",
        train_mean * 100.0,
        train_std * 100.0
    );
    println!(
        "Test  Accuracy = {:.2}% ± {:.2}%",
        test_mean * 100.0,
        test_std * 100.0
    );

    Ok(())
}
