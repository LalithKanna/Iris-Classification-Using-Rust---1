use crate::model::NeuralNetwork;
use crate::data::TrainingSample;

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

pub fn confusion_matrix(
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