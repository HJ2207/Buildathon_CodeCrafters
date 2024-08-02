

// document.getElementById('fileInput').addEventListener('change', (event) => {
//     const file = event.target.files[0];

//     if (file) {
//         const reader = new FileReader();

//         reader.onload = function(e) {
//             const fileContent = e.target.result;
//             ipcRenderer.send('upload-file', file.name, fileContent);
//         };

//         reader.readAsText(file);
//     }
// });

// Example file path
const { ipcRenderer } = require('electron');


function ingestData() {
    const fileInput = document.getElementById('file');
    document.getElementById("box").innerHTML = "hi1";
    const file = fileInput.files[0];
    document.getElementById("box").innerHTML = "hi2";
    const path = file.path;
    document.getElementById("box").innerHTML = "hi3";
    const filename = file.name;

    document.getElementById("box").innerHTML = "hi4";
    console.log(`Filename: ${filename}`);
    document.getElementById("box").innerHTML = "hi5";

    ipcRenderer.send('ingest-data', { path, filename });
    document.getElementById("box").innerHTML = "hi6";
}

ipcRenderer.on('confirmation-message', (event, message) => {
    const confirmationDiv = document.getElementById('remove-features-confirmation');
    confirmationDiv.style.display = 'block';
    confirmationDiv.innerHTML = message;
});


// const { ipcRenderer } = require('electron');

// function uploadFile() {
//     const fileInput = document.getElementById('filename');
//     const file = fileInput.files[0];
//     document.getElementById("box").innerHTML = "hi1";

//     if (file) {
//         const reader = new FileReader();
//         reader.onload = function () {
//             const fileContent = reader.result;
//             ipcRenderer.send('file-upload', { filename: file.name, content: fileContent });
//         };
//         reader.readAsArrayBuffer(file);
//     } else {
//         console.error('No file selected.');
//     }
//     document.getElementById("box").innerHTML = "hi";
// }


// // Ingest Data
// function ingestData() {
//   const path = document.getElementById('path').value;
//   const filename = document.getElementById('filename').value;
//   document.getElementById("box").innerHTML = path;
//   console.log(filename);
//   ipcRenderer.send('ingest-data', { path, filename });
// }

// Show Data Dimensions
function showDataDimensions() {
    const path = document.getElementById('path').value;
    ipcRenderer.send('show-data-dimensions');
}

// Remove Features
function removeFeatures() {
  const removeFeatures = document.getElementById('remove-features').value;
  ipcRenderer.send('remove-features', { removeFeatures });
}

// Convert to Numbers
function convertNumbers() {
  const convertNumbers = document.getElementById('convert-numbers').value;
  ipcRenderer.send('convert-numbers', { convertNumbers });
}

// Train Test Split
function trainTestSplit() {
  const trainData = document.getElementById('traindata').value;
  const testData = document.getElementById('testdata').value;

  ipcRenderer.send('train-test-split', { trainData, testData });
}

// Train Random Forest
function trainRandomForest() {
  const criterion = document.getElementById('criterion').value;
  const maxDepth = document.getElementById('max-depth').value;
  const nEstimators = document.getElementById('N-estimators').value;

  ipcRenderer.send('train-random-forest', { criterion, maxDepth, nEstimators });
}

// Listen for the result from the main process
ipcRenderer.on('python-result', (event, result) => {
  console.log('Result from Python:', result);
  // Handle the result as needed (e.g., display it to the user)
});

// Update confirmation message
ipcRenderer.on('confirmation-message', (event, message) => {
  document.getElementById('confirmation-message').textContent = message;
});

// Listen for data dimensions result
ipcRenderer.on('data-dimensions', (event, dimensions) => {
  document.getElementById('num-rows-value').textContent = dimensions.numRows;
  document.getElementById('num-columns-value').textContent = dimensions.numColumns;
});


// Train Random Forest
function trainRandomForest() {
  const criterion = document.getElementById('criterion').value;
  const maxDepth = document.getElementById('max-depth').value;
  const nEstimators = document.getElementById('N-estimators').value;

  ipcRenderer.send('train-random-forest', { criterion, maxDepth, nEstimators });
}

// Listen for the result from the main process
ipcRenderer.on('python-result', (event, result) => {
  console.log('Result from Python:', result);
  // Handle the result as needed (e.g., display it to the user)
});




// const remo = document.getElementById('remove-features');
// const conv = document.getElementById('convert-numbers');
// const tar = document.getElementById('target');


// // Handle form submission
// form.addEventListener('submit', (event) => {
//     event.preventDefault(); // Prevent the default form submission

//     // Gather input values
//     const inputData = {
//         remove: remo.value,
//         convert: conv.value,
//         target: tar.checked
//     };

    

//     // Convert input data to JSON string
//     const jsonString = JSON.stringify(inputData);

//     // Send the JSON string to the main process
//     ipcRenderer.send('send-data', jsonString);
// });

// // Listen for the result from the main process
// ipcRenderer.on('python-result', (event, result) => {
//     console.log('Result from Python:', result);
//     // Handle the result as needed (e.g., display it to the user)
// });




const { ipcRenderer } = require('electron');



ipcRenderer.on('confirmation-message', (event, message) => {
    document.querySelectorAll('.confirmation-message').forEach(el => {
        el.textContent = message;
        el.style.display = 'block';
    });
});

ipcRenderer.on('data-dimensions', (event, dimensions) => {
    document.getElementById('num-rows').textContent = `Number of rows: ${dimensions.numRows}`;
    document.getElementById('num-columns').textContent = `Number of columns: ${dimensions.numColumns}`;
});

ipcRenderer.on('performance', (event, result) => {
    document.getElementById(`${result.modelType}-performance`).textContent = result.performance;
});
