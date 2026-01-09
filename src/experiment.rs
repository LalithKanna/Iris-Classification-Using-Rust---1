use crate::data::{Sample, shuffle_and_split, compute_mean_std, normalize_dataset, to_training_samples};
use crate::model::{NeuralNetwork, save_model};
use crate::training::{accuracy, confusion_matrix};
use crate::utils::{save_confusion_matrix, save_curve};

pub struct ExperimentResult {
    pub train_acc: f64,
    pub test_acc: f64,
    pub losses: Vec<f64>,
    pub train_accs: Vec<f64>,
    pub test_accs: Vec<f64>,
    pub confusion_matrix: Vec<Vec<usize>>,
}

pub fn run_experiment(dataset: &Vec<Sample>) -> ExperimentResult {
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

pub fn save_experiment_results(result: &ExperimentResult) {
    save_confusion_matrix(&result.confusion_matrix, "confusion_matrix.csv");
    save_curve(&result.losses, "loss_curve.csv");
    save_curve(&result.train_accs, "train_accuracy.csv");
    save_curve(&result.test_accs, "test_accuracy.csv");
}