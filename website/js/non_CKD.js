window.onload = function () {
  fetch('http://127.0.0.1:8000/static/temp/important_features.json')
    .then(response => response.json())
    .then(data => {
      const table = document.createElement('table');
      table.border = '1';
      table.style.borderCollapse = 'collapse';
      table.style.marginTop = '20px';

      const header = document.createElement('tr');
      header.innerHTML = '<th style="padding: 8px;">Feature</th><th style="padding: 8px;">Value</th><th style="padding: 8px;">Normal Range</th>';
      table.appendChild(header);

      // Define ranges for scaling
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

      // Function to scale the value back to the original range
      function reverseScale(feature, value) {
          const range = ranges[feature.toLowerCase()];
          if (range) {
              // Reverse the scaling based on the formula (scaled_value * (max - min) + min)
              return (value * (range[1] - range[0])) + range[0];
          }
          return value; // If no range found, return the value as is
      }

      // Loop through each feature in the data
      for (const [key, value] of Object.entries(data)) {
        const row = document.createElement('tr');

        const feature = document.createElement('td');
        feature.innerText = key;

        const val = document.createElement('td');
        const originalValue = reverseScale(key, value); // Convert scaled value back to original
        val.innerText = originalValue.toFixed(2); // Show with two decimal places
        
        // Get the normal range for the feature
        const range = ranges[key.toLowerCase()];
        const rangeText = range ? `${range[0]} - ${range[1]}` : 'N/A';

        const normalRange = document.createElement('td');
        normalRange.innerText = rangeText;

        row.appendChild(feature);
        row.appendChild(val);
        row.appendChild(normalRange); // Append normal range to the row
        table.appendChild(row);
      }

      document.getElementById('feature-table').appendChild(table);
    })
    .catch(err => {
      console.error('Could not load feature data', err);
      document.getElementById('feature-table').innerText = 'Failed to load important features.';
    });
};