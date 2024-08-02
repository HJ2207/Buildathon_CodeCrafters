document.getElementById('model').addEventListener('change', (event) => {
    const selectedModel = event.target.value;
    document.querySelectorAll('.model-params').forEach((elem) => {
      elem.style.display = 'none';
    });
    if (selectedModel === 'random_forest') {
      document.getElementById('random_forest_params').style.display = 'block';
    } else if (selectedModel === 'adaboost') {
      document.getElementById('adaboost_params').style.display = 'block';
    }
  });
  
  document.getElementById('ml-form').addEventListener('submit', (event) => {
    event.preventDefault();
  
    const file = document.getElementById('file-input').files[0];
    const model = document.getElementById('model').value;
    const maxDepth = document.getElementById('max_depth').value;
    const nEstimators = document.getElementById('n_estimators').value;
  
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    if (maxDepth) {
      formData.append('max_depth', maxDepth);
    }
    if (nEstimators) {
      formData.append('n_estimators', nEstimators);
    }
  
    fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        const { exec } = require('child_process');
        const fs = require('fs');
        const ejs = require('ejs');
        
        ejs.renderFile('index.ejs', { results: data }, {}, (err, str) => {
          if (err) {
            console.error(err);
            return;
          }
          fs.writeFileSync('index.html', str);
          exec('npx electron .');
        });
      });
  });
  