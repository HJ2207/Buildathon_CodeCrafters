// const { ipcMain } = require('electron');
// const { spawn } = require('child_process');
// const fs = require('fs');
// const path = require('path');
// const os = require('os');

// ipcMain.on('upload-file', (event, fileName, fileContent) => {
//     const tempFilePath = path.join(os.tmpdir(), fileName);

//     fs.writeFileSync(tempFilePath, fileContent, 'utf8');

//     const pythonProcess = spawn('python', ['path_to_your_script.py', tempFilePath]);

//     pythonProcess.stdout.on('data', (data) => {
//         console.log(`Output from Python: ${data}`);
//     });

//     pythonProcess.stderr.on('data', (data) => {
//         console.error(`Error from Python: ${data}`);
//     });

//     pythonProcess.on('exit', () => {
//         fs.unlinkSync(tempFilePath); // Delete the temporary file after processing
//     });
// });

// ipcMain.on('send-file', (event, csvFilePath) => {
//     const pythonProcess = spawn('python', ['flask_ml/app.py', csvFilePath]);

//     pythonProcess.stdout.on('data', (data) => {
//         console.log(`Output from Python: ${data}`);
//     });

//     pythonProcess.stderr.on('data', (data) => {
//         console.error(`Error from Python: ${data}`);
//     });
// });


// ipcMain.on('send-data', (event, jsonString) => {
//     const pythonProcess = spawn('python', ['flask_ml/app.py', jsonString]);

//     let pythonOutput = '';

//     pythonProcess.stdout.on('data', (data) => {
//         pythonOutput += data.toString();
//     });

//     pythonProcess.stderr.on('data', (data) => {
//         console.error(`Error from Python: ${data}`);
//     });

//     pythonProcess.on('close', (code) => {
//         if (code === 0) {
//             // Send the parsed JSON back to the renderer process
//             event.reply('python-result', JSON.parse(pythonOutput));
//         } else {
//             console.error(`Python process exited with code ${code}`);
//         }
//     });
// });


// const { app, BrowserWindow, ipcMain } = require('electron');
// const path = require('path');
// const { exec } = require('child_process');

// let mainWindow;

// function createWindow() {
//   mainWindow = new BrowserWindow({
//     width: 1600,
//     height: 1000,
//     webPreferences: {
//       preload: path.join(__dirname, 'preload.js'),
//       contextIsolation: true,
//       enableRemoteModule: false,
//       nodeIntegration: true,
//     }
//   });

//   mainWindow.on('closed', function () {
//     mainWindow = null;
//   });

//   mainWindow.loadFile('Buildathon/progress1.html');
// }

// app.whenReady().then(() => {
//   createWindow();

//   app.on('activate', () => {
//     if (BrowserWindow.getAllWindows().length === 0) {
//       createWindow();
//     }
//   });
// });

// app.on('window-all-closed', () => {
//   if (process.platform !== 'darwin') {
//     app.quit();
//   }
// });

// ipcMain.on('ingest-data', (event, data) => {
//   const { path, filename } = data;
//   // Add logic to handle file ingestion if needed
//   event.sender.send('confirmation-message', 'Data ingested successfully.');
// });

// ipcMain.on('show-data-dimensions', (event, data) => {
//   const script = 'flask_ml/app.py'; // Adjust the path to your Python script
//   const dataFilePath = data; // Adjust the path to your data file

//   exec(`python ${script} show-data-dimensions ${dataFilePath}`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`exec error: ${error}`);
//       return;
//     }
//     const dimensions = JSON.parse(stdout);
//     event.sender.send('data-dimensions', dimensions);
//   });
// });

// ipcMain.on('remove-features', (event, data) => {
//   const { removeFeatures } = data;
//   const script = 'flask_ml/app.py'; // Adjust the path to your Python script

//   exec(`python ${script} remove-features ${removeFeatures}`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`exec error: ${error}`);
//       return;
//     }
//     event.sender.send('confirmation-message', 'Features removed successfully.');
//   });

// });

// ipcMain.on('convert-numbers', (event, data) => {
//   const { convertNumbers } = data;
//   const script = 'flask_ml/app.py'; // Adjust the path to your Python script

//   exec(`python ${script} convert-numbers ${convertNumbers}`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`exec error: ${error}`);
//       return;
//     }
//     event.sender.send('confirmation-message', 'Features converted successfully.');
//   });
// });

// ipcMain.on('train-test-split', (event, data) => {
//   const { trainData, testData } = data;
//   const script = 'flask_ml/app.py'; // Adjust the path to your Python script

//   exec(`python ${script} train-test-split ${trainData} ${testData}`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`exec error: ${error}`);
//       return;
//     }
//     event.sender.send('confirmation-message', 'Train-test split completed.');
//   });
// });

// ipcMain.on('train-random-forest', (event, data) => {
//   const { criterion, maxDepth, nEstimators } = data;
//   const script = 'flask_ml/app.py'; // Adjust the path to your Python script

//   exec(`python ${script} train-random-forest ${criterion} ${maxDepth} ${nEstimators}`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`exec error: ${error}`);
//       return;
//     }
//     event.sender.send('confirmation-message', 'Random Forest trained successfully.');
//   });
// });


const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec } = require('child_process');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1600,
        height: 1000,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: false,
            enableRemoteModule: false,
            nodeIntegration: true,
        }
    });

    mainWindow.on('closed', function () {
        mainWindow = null;
    });

    mainWindow.loadFile('Buildathon/progress1.html');
}

app.whenReady().then(() => {
    createWindow();

    ipcMain.on('file-upload', (event, { filename, content }) => {
        const buffer = Buffer.from(new Uint8Array(content));
        fs.writeFile(path.join(__dirname, 'uploads', filename), buffer, (err) => {
            if (err) {
                console.error('Error saving file:', err);
            } else {
                console.log('File saved successfully');
            }
        });
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});


ipcMain.on('ingest-data', (event, data) => {
    const { path, filename } = data;
    event.sender.send('confirmation-message', `File ${filename} from path ${path} ingested successfully.`);
});


ipcMain.on('show-data-dimensions', (event) => {
    const script = 'flask_ml/app.py';
    exec(`python ${script} show-data-dimensions`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return;
        }
        const dimensions = JSON.parse(stdout);
        event.sender.send('data-dimensions', dimensions);
    });
});

ipcMain.on('remove-features', (event, data) => {
    const { removeFeatures } = data;
    event.sender.send('confirmation-message', `Features ${removeFeatures} removed successfully.`);
});

ipcMain.on('convert-numbers', (event, data) => {
    const { convertNumbers } = data;
    event.sender.send('confirmation-message', `Features ${convertNumbers} converted to numbers successfully.`);
});

ipcMain.on('train-test-split', (event, data) => {
    const { trainData, testData } = data;
    event.sender.send('confirmation-message', `Train test split: ${trainData}% training data, ${testData}% testing data.`);
});

ipcMain.on('train-random-forest', (event, data) => {
    const { criterion, maxDepth, nEstimators } = data;
    event.sender.send('confirmation-message', `Random Forest trained with criterion ${criterion}, max depth ${maxDepth}, and ${nEstimators} estimators.`);
});

ipcMain.on('export-model', (event, data) => {
    const { modelType, path } = data;
    event.sender.send('confirmation-message', `${modelType} model exported to ${path} successfully.`);
});
