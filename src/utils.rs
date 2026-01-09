use std::fs::File;
use std::io::Write;

pub fn mean(values: &Vec<f64>) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn std(values: &Vec<f64>, mean: f64) -> f64 {
    let variance = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

pub fn save_confusion_matrix(matrix: &Vec<Vec<usize>>, path: &str) {
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

pub fn save_curve(values: &Vec<f64>, path: &str) {
    let mut file = File::create(path).unwrap();
    for (epoch, val) in values.iter().enumerate() {
        writeln!(file, "{},{}", epoch, val).unwrap();
    }
}

pub fn save_values(values: &Vec<f64>, path: &str) {
    let mut file = File::create(path).unwrap();
    for v in values {
        writeln!(file, "{}", v).unwrap();
    }
}

pub fn save_summary(
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