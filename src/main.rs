mod data;
mod model;
mod training;
mod utils;
mod experiment;

use std::error::Error;
use crate::data::load_iris;
use crate::experiment::{run_experiment, save_experiment_results};
use crate::utils::{mean, std, save_summary, save_values};

fn main() -> Result<(), Box<dyn Error>> {
    let dataset = load_iris("data/iris.csv")?;
    assert_eq!(dataset.len(), 150);

    let runs = 30;

    let mut train_accuracies = Vec::new();
    let mut test_accuracies = Vec::new();

    // Run ONE experiment for plots
    let first_run = run_experiment(&dataset);
    save_experiment_results(&first_run);

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