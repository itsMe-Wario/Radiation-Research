const inputIds = [
    "glucose", "cholesterol", "hemoglobin", "wbc", "rbc",
    "hematocrit", "insulin", "bmi", "sbp", "dbp",
    "triglycerides", "hba1c", "ldlc", "hdlc", "alt", "ast", "hr",
    "creatinine", "troponin", "crp"
];

const ranges = {
    glucose: [70, 140],
    cholesterol: [125, 200],
    hemoglobin: [13.5, 17.5],
    wbc: [4000, 11000],
    rbc: [4.2, 5.4],
    hematocrit: [38, 52],
    insulin: [5, 25],
    bmi: [18.5, 24.9],
    sbp: [90, 120],
    dbp: [60, 80],
    triglycerides: [50, 150],
    hba1c: [4, 6],
    ldlc: [70, 130],
    hdlc: [40, 60],
    alt: [10, 40],
    ast: [10, 40],
    hr: [60, 100],
    creatinine: [0.6, 1.2],
    troponin: [0, 0.04],
    crp: [0, 3]
};

function normalize(value, [min, max]) {
    return (value - min) / (max - min);
}

document.getElementById("customFileButton").addEventListener("click", function () {
    document.getElementById("csvFile").click();
});

document.getElementById("csvFile").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async function (e) {
        try {
            const jsonData = JSON.parse(e.target.result);

            let rawValues = {};
            for (let id of inputIds) {
                if (!(id in jsonData)) {
                    alert(`Missing field "${id}" in JSON file.`);
                    return;
                }
                rawValues[id] = parseFloat(jsonData[id]);
            }

            const features = inputIds.map(id => normalize(rawValues[id], ranges[id]));

            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ features })
            });

            const result = await response.json();
            alert(`Predicted disease: ${result.predicted_disease}`);
        } catch (error) {
            console.error("Error reading or parsing JSON:", error.message);
            console.error("Error stack trace:", error.stack);
            alert(`An error occurred: ${error.message}. Please check the console for more details.`);
        }
        
    };

    reader.readAsText(file);

    const fileName = file.name;
    document.getElementById("output").textContent = `Selected file: ${fileName}`;
});