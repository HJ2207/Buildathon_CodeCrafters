// import express from 'express';
// import bodyParser from 'body-parser';
// import pg from 'pg';
// import bcrypt from 'bcrypt';
// import session from 'express-session';
// import dotenv from 'dotenv';
// import path from 'path';
// // const express = require('express');
// // const bodyParser = require('body-parser');
// // const pg = require('pg');
// // const bcrypt = require('bcrypt');
// // const session = require('express-session');
// // const dotenv = require('dotenv');
// // const path = require('path');


// const app = express();
// const saltRounds = 10;
// dotenv.config();
// app.use(
//   session({
//     secret: process.env.SESSION_SECRET,
//     resave: false,
//     saveUninitialized: true,
//   })
// );
// app.use(bodyParser.urlencoded({ extended: true }));

// const db = new pg.Client({
//   user: process.env.PG_USER,
//   host: process.env.PG_HOST,
//   database: process.env.PG_DATABASE,
//   password: process.env.PG_PASSWORD,
//   port: process.env.PG_PORT,
// });
// db.connect();

// app.get("/", (req, res) => {
//   res.sendFile(path.join(__dirname, "..","Buildathon", "index.html"));
// });

// app.get("/signup", (req, res) => {
//   res.sendFile(path.join(__dirname, "..","Buildathon", "signup.html"));
// });

// //login
// app.post("/",async (req,res)=>{
//   const email = req.body.username;
//   const loginPassword = req.body.password;

//   try {
//     const result = await db.query("SELECT * FROM auth WHERE username = $1", [
//       email,
//     ]);
//     if (result.rows.length > 0) {
//       const user = result.rows[0];
//       const storedHashedPassword = user.password;
//       //verifying the password
//       bcrypt.compare(loginPassword, storedHashedPassword, (err, result) => {
//         if (err) {
//           console.error("Error comparing passwords:", err);
//         } else {
//           if (result) {
//             res.sendFile(path.join(__dirname, "..", "Buildathon", "pages.html"));
//           } else {
//             res.status(401).send("Incorrect password");
//           }
//         }
//       });
//     } else {
//       res.status(404).send("User not found");
//     }
//   } catch (err) {
//     console.error("Error during login:", err);
//     res.status(500).send("Internal server error");
//   }
// });

// app.post("/signup",async (req,res)=>{
//   const name = req.body.username;
//   const password = req.body.password;

//   try {
//     const checkResult = await db.query("SELECT * FROM auth WHERE username = $1", [
//       name,
//     ]);

//     if (checkResult.rows.length > 0) {
//       req.redirect("/");
//     } else {
//       bcrypt.hash(password, saltRounds, async (err, hash) => {
//         if (err) {
//           console.error("Error hashing password:", err);
//           res.status(500).send("Internal server error");
//         } else {
//           try {
//             const result = await db.query(
//               "INSERT INTO auth (username, password) VALUES ($1, $2) RETURNING *",
//               [name, hash]
//             );
//             const user = result.rows[0];
//             console.log("User name:", user);
//             res.sendFile(path.join(__dirname, "..", "Buildathon", "pages.html"));
//           } catch (err) {
//             console.error("Error inserting user into the database:", err);
//             res.status(500).send("Internal server error")
//           }
//         }
//       });
//     }
//   } catch (err) {
//     console.log(err);
//     res.status(500).send("Internal server error")
//   }
// });

// export default app;
