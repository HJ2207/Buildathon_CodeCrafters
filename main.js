// const { app, BrowserWindow } = require('electron');

// function createWindow() {
//     const win = new BrowserWindow({
//         width: 800,
//         height: 600,
//         webPreferences: {
//             nodeIntegration: true
//         }
//     });

//     win.loadFile('index.html');
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

// Import required modules
// main.js
const { app, BrowserWindow, ipcMain } = require('electron');
const bcrypt = require('bcrypt');
const pg = require('pg');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// Create a new PostgreSQL client
const db = new pg.Client({
  user: process.env.PG_USER,
  host: process.env.PG_HOST,
  database: process.env.PG_DATABASE,
  password: process.env.PG_PASSWORD,
  port: process.env.PG_PORT,
});
db.connect();

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1000,
    height: 1000,
    webPreferences: {
      contextIsolation: false,
      nodeIntegration: true,
    },
  });

  win.loadFile('index.html');
};

app.whenReady().then(createWindow);

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

// Handle login form submission
ipcMain.on('login', (event, username, password) => {
  // Query the database to check if the user exists
  db.query("SELECT * FROM auth WHERE username = $1", [username], (err, result) => {
    if (err) {
      console.error("Error during login:", err);
      event.reply('login-error', 'Internal server error');
    } else {
      if (result.rows.length > 0) {
        const user = result.rows[0];
        const storedHashedPassword = user.password;
        // Verify the password
        bcrypt.compare(password, storedHashedPassword, (err, result) => {
          if (err) {
            console.error("Error comparing passwrds:", err);
            event.reply('login-error', 'Internal server error');
          } else {
            if (result) {
              event.reply('login-success', 'Logged in successfully!');
              // Load the pages.html file
              win.loadURL(`file://${__dirname}/pages.html`);
            } else {
              event.reply('login-error', 'Incorrect password');
            }
          }
        });
      } else {
        event.reply('login-error', 'User not found');
      }
    }
  });
});

// Handle signup form submission
ipcMain.on('signup', (event, username, password) => {
  // Check if the user already exists
  db.query("SELECT * FROM auth WHERE username = $1", [username], (err, result) => {
    if (err) {
      console.error("Error during signup:", err);
      event.reply('signup-error', 'Internal server error');
    } else {
      if (result.rows.length > 0) {
        event.reply('signup-error', 'User already exists');
      } else {
        // Hash the password
        bcrypt.hash(password, 10, (err, hash) => {
          if (err) {
            console.error("Error hashing password:", err);
            event.reply('signup-error', 'Internal server error');
          } else {
            // Insert the user into the database
            db.query(
              "INSERT INTO auth (username, passwrd) VALUES ($1, $2) RETURNING *",
              [username, hash],
              (err, result) => {
                if (err) {
                  console.error("Error inserting user into the database:", err);
                  event.reply('signup-error', 'Internal server error');
                } else {
                  event.reply('signup-success', 'Signed up successfully!');
                  // Load the pages.html file
                  win.loadURL(`file://${__dirname}/pages.html`);
                }
              }
            );
          }
        });
      }
    }
  });
});
