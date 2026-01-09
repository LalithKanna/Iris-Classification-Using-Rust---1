import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Create output directory
# -----------------------------
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Global plotting style
# -----------------------------
sns.set(style="whitegrid", font_scale=1.1)

# -----------------------------
# 1. Confusion Matrix
# -----------------------------
cm = pd.read_csv("confusion_matrix.csv", header=None)
class_names = ["Setosa", "Versicolor", "Virginica"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------
# 2. Loss vs Epoch
# -----------------------------
loss = pd.read_csv("loss_curve.csv", header=None, names=["Epoch", "Loss"])

plt.figure(figsize=(7, 5))
plt.plot(loss["Epoch"], loss["Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=300)
plt.show()

# -----------------------------
# 3. Accuracy vs Epoch
# -----------------------------
train_acc = pd.read_csv("train_accuracy.csv", header=None, names=["Epoch", "Accuracy"])
test_acc = pd.read_csv("test_accuracy.csv", header=None, names=["Epoch", "Accuracy"])

plt.figure(figsize=(7, 5))
plt.plot(train_acc["Epoch"], train_acc["Accuracy"], label="Train Accuracy")
plt.plot(test_acc["Epoch"], test_acc["Accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_vs_epoch.png", dpi=300)
plt.show()

# -----------------------------
# 4. Test Accuracy Distribution (Histogram)
# -----------------------------
acc_dist = pd.read_csv(
    "test_accuracy_distribution.csv", header=None, names=["Accuracy"]
)

plt.figure(figsize=(6, 5))
plt.hist(acc_dist["Accuracy"], bins=8, edgecolor="black")
plt.xlabel("Test Accuracy")
plt.ylabel("Frequency")
plt.title("Distribution of Test Accuracy Across Runs")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/test_accuracy_histogram.png", dpi=300)
plt.show()

# -----------------------------
# 5. Test Accuracy Boxplot
# -----------------------------
plt.figure(figsize=(4, 5))
plt.boxplot(acc_dist["Accuracy"], vert=True)
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Boxplot")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/test_accuracy_boxplot.png", dpi=300)
plt.show()

# -----------------------------
# 6. Mean ± Std Summary Table
# -----------------------------
summary = pd.read_csv("summary.csv")

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis("off")

table = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    loc="center",
    cellLoc="center",
    colLoc="center",
)

# Improve table appearance
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.6)

# Bold header row
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#EAEAEA")

plt.title("Performance Summary (Mean ± Std)", pad=12, fontsize=14)
plt.tight_layout()
plt.savefig("plots/summary_table.png", dpi=300)
plt.show()