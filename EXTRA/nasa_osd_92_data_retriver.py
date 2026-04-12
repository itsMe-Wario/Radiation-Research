import requests
from pathlib import Path
import json

#Configuration
DATASET_ID = "OSD-92"
ASSAY_ID = "OSD-92_transcription-profiling_dna-microarray_Agilent"

BASE_API = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset"
BASE_DIR = Path("\\Studies\\Aplicable Studies\\Radiation in a Human Three-Dimensional Tissue Model\\OSD-92_data")

BASE_DIR.mkdir(exist_ok=True)

#File downloader
def download_file(url: str, destination: Path):
    if destination.exists():
        print(f"[SKIP] {destination.name}")
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)

    print(f"[DOWNLOADED] {destination.name}")

# Step 1: Query study-level files
study_files_url = f"{BASE_API}/{DATASET_ID}/files/"
study_files = requests.get(study_files_url, timeout=30).json()

# Step 2: Query assay-level files
assay_files_url = f"{BASE_API}/{DATASET_ID}/assay/{ASSAY_ID}/files/"
assay_files = requests.get(assay_files_url, timeout=30).json()

# Step 3: Normalize file inventory
manifest = []

# Study-level files
for filename, info in study_files[DATASET_ID]["files"].items():
    manifest.append({
        "filename": filename,
        "url": info["URL"],
        "scope": "study"
    })

# Assay-level files
assay_files_dict = (
    assay_files[DATASET_ID]["assays"][ASSAY_ID]["files"]
)

for filename, info in assay_files_dict.items():
    manifest.append({
        "filename": filename,
        "url": info["URL"],
        "scope": "assay"
    })

# Deduplicate files (study == assay duplicates)
unique_manifest = {
    item["filename"]: item for item in manifest
}.values()

# Save manifest for reproducibility
with open(BASE_DIR / "file_manifest.json", "w") as fh:
    json.dump(list(unique_manifest), fh, indent=2)

# Step 4: Compartmentalization logic
def categorize_file(filename: str) -> str:
    """
    Assigns files to logical directories based on filename patterns.
    """
    if filename.endswith(".zip"):
        return "archives"
    if "normalized_expression" in filename:
        return "processed/normalized_expression"
    if "differential_expression" in filename:
        return "processed/differential_expression"
    if "PCA" in filename:
        return "processed/visualization"
    if "SampleTable" in filename:
        return "metadata"
    if "raw_intensities" in filename:
        return "raw/intensities"
    if filename.endswith(".txt") and "GSM" in filename:
        return "raw/GSM_microarray_files"
    return "misc"

# Step 5: Download and organize
for item in unique_manifest:
    category = categorize_file(item["filename"])
    target_dir = BASE_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)

    destination = target_dir / item["filename"]
    download_file(item["url"], destination)

print("All files processed successfully.")