document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("csvFile");
    const fileButton = document.getElementById("customFileButton");
    const output = document.getElementById("output");

    fileButton.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        fetch("http://127.0.0.1:3000/predict", {
            method: "POST",
            body: formData,
        })
    });
});

