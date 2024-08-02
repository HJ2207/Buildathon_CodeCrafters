document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    if (username === '' || password === '') {
        alert('Please fill in both fields.');
        return;
    }

    // You can replace the following lines with your own login logic
    console.log('Username:', username);
    console.log('Password:', password);

    // Example: redirecting to another page after login
    window.location.href = 'pages.html';
    
});
