// const { app, BrowserWindow } = require('electron');

// function createWindow() {
//     const win = new BrowserWindow({
//         width: 800,
//         height: 600,
//         webPreferences: {
//             nodeIntegration: true
//         }
//     });

//     win.loadFile('SignIn.html');
// }

// app.whenReady().then(createWindow);

// app.on('window-all-closed', () => {
//     if (process.platform !== 'darwin') {
//         app.quit();
//     }
// });

// app.on('activate', () => {
//     if (BrowserWindow.getAllWindows().length === 0) {
//         createWindow();
//     }
// });

const { app, BrowserWindow } = require('electron');
const expressApp = require('./authentication/auth').default;
const path = require('path');
const fs = require('fs');
const ejs = require('ejs');

function createWindow() {
  win = new BrowserWindow({
    width: 1600,
    height: 1000,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      enableRemoteModule: false,
    },
  });

  win.on('closed', function () {
    win = null;
  });

  // Render EJS template with initial data
  ejs.renderFile('index.ejs', { results: null }, {}, (err, str) => {
    if (err) {
      console.error(err);
      return;
    }
    fs.writeFileSync('index.html', str);
    win.loadFile('index.html');
  });
}
async function main() {
  const expressApp = (await import('./authentication/indexApp.js')).default;

  app.whenReady().then(() => {
    const server = expressApp.listen(3000, () => {
      console.log('Express server listening on port 3000');
      createWindow();
    });
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
      }
    });
  });
  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });
}

main().catch(console.error);
