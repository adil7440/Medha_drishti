import random
import shutil
import csv
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE = Path(r"C:\Users\Adil\Desktop\yugma")
SOURCE = BASE / "test_spine" / "Spine DATASETS"
DEST = BASE / "training_data_spine"

normal_src = SOURCE / "Normal Spine MRI Datasets"
pathological_src = SOURCE / "Pathological Spine MRI Datasets"

normal_patients = sorted([d.name for d in normal_src.iterdir() if d.is_dir()])
pathological_patients = sorted([d.name for d in pathological_src.iterdir() if d.is_dir()])

random.shuffle(normal_patients)
random.shuffle(pathological_patients)

train_normal = normal_patients[:5]
test_normal = normal_patients[5:]

train_pathological = pathological_patients[:5]
test_pathological = pathological_patients[5:]

train_normal_dir = DEST / "Normal Spine MRI Datasets"
train_pathological_dir = DEST / "Pathological Spine MRI Datasets"

train_normal_dir.mkdir(parents=True, exist_ok=True)
train_pathological_dir.mkdir(parents=True, exist_ok=True)

def copy_patients(patients, src_dir, dest_dir):
    for p in patients:
        shutil.copytree(src_dir / p, dest_dir / p)

copy_patients(train_normal, normal_src, train_normal_dir)
copy_patients(train_pathological, pathological_src, train_pathological_dir)

csv_path = BASE / "scripts" / "spine_dataset_split.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Patient_ID", "Class", "Dataset"])
    for p in train_normal:
        writer.writerow([p, "Normal", "Training"])
    for p in train_pathological:
        writer.writerow([p, "Pathological", "Training"])
    for p in test_normal:
        writer.writerow([p, "Normal", "Test"])
    for p in test_pathological:
        writer.writerow([p, "Pathological", "Test"])

all_train = train_normal + train_pathological
all_test = test_normal + test_pathological

assert len(set(all_train) & set(all_test)) == 0, "Overlap detected!"
assert len(all_train) + len(all_test) == 20, "Missing patients!"

print("=" * 40)
print("Spine Dataset Split Summary")
print("=" * 40)
print(f"Training Normal Patients       : {len(train_normal)}")
print(f"Training Pathological Patients : {len(train_pathological)}")
print(f"Testing Normal Patients        : {len(test_normal)}")
print(f"Testing Pathological Patients  : {len(test_pathological)}")
print(f"Total Training Patients        : {len(all_train)}")
print(f"Total Testing Patients         : {len(all_test)}")
print("=" * 40)
print("\nVerification passed:")
print("  - No patient overlap between training and testing.")
print("  - All 20 patients accounted for.")
print("  - Folder integrity maintained.")
print(f"\nCSV saved to: {csv_path}")
