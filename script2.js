function showSection(sectionId) {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    document.getElementById(sectionId).style.display = 'block';
}

function ingestData() {
    const path = document.getElementById('path').value;
    const filename = document.getElementById('filename').value;
    document.getElementsByClassName('confirmation-message').innerText = `File ${filename} from path ${path} ingested successfully.`;
}

function showDataDimensions() {
    document.getElementById('num-rows').innerText = 100; // Example value
    document.getElementById('num-columns').innerText = 10; // Example value
}

function trainTestSplit() {
    const trainSplit = document.getElementById('train-split').value;
    const testSplit = document.getElementById('test-split').value;
    alert(`Data split into ${trainSplit}% training and ${testSplit}% testing.`);
}


function trainRandomForest() {
    const criterion = document.getElementById('criterion').value;
    const maxDepth = document.getElementById('max-depth').value;
    const nEstimators = document.getElementById('n-estimators').value;
    document.getElementById('rf-performance').innerText = `Random Forest trained with ${criterion}, max depth ${maxDepth}, and ${nEstimators} estimators.`;
}

function trainDecisionTree() {
    const criterion = document.getElementById('criterion').value;
    const maxDepth = document.getElementById('max-depth').value;
    const nEstimators = document.getElementById('n-estimators').value;
    document.getElementById('DT-performance').innerText = `Decision Tree trained with ${criterion}, max depth ${maxDepth}, and ${nEstimators} estimators.`;
}

function trainSVM() {
    const criterion = document.getElementById('criterion').value;
    const maxDepth = document.getElementById('max-depth').value;
    const nEstimators = document.getElementById('n-estimators').value;
    document.getElementById('svm-performance').innerText = `SVM trained with ${criterion}, max depth ${maxDepth}, and ${nEstimators} estimators.`;
}

function trainXGBoost() {
    const criterion = document.getElementById('criterion').value;
    const maxDepth = document.getElementById('max-depth').value;
    const nEstimators = document.getElementById('n-estimators').value;
    document.getElementById('XG-performance').innerText = `XG-Boost trained with ${criterion}, max depth ${maxDepth}, and ${nEstimators} estimators.`;
}

function trainADABoost() {
    const criterion = document.getElementById('ada-criterion').value;
    const maxDepth = document.getElementById('ada-max-depth').value;
    const nEstimators = document.getElementById('ada-n-estimators').value;
    document.getElementById('ada-performance').innerText = `ADA Boosting trained with ${criterion}, max depth ${maxDepth}, and ${nEstimators} estimators.`;
}

function exportModel(modelType) {
    const path = document.getElementById(`${modelType}-path`).value;
    document.getElementById(`${modelType}-confirmation`).innerText = `Model exported to ${path} successfully.`;
}
function showConfirmationMessage() {
    var confirmationMessage = document.getElementsByClassName('confirmation-message');
    confirmationMessage.style.display = 'block';
    // setTimeout(function() {
    //     confirmationMessage.style.display = 'none';
    // }, 3000); // Hide the message after 3 seconds
}
