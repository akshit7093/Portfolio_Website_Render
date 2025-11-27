import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Security: Disable Right Click
document.addEventListener('contextmenu', (e) => e.preventDefault());

// Security: Disable Keyboard Shortcuts (F12, Ctrl+Shift+I, etc.)
document.addEventListener('keydown', (e) => {
    if (
        e.key === 'F12' ||
        (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'J' || e.key === 'C')) ||
        (e.ctrlKey && e.key === 'U')
    ) {
        e.preventDefault();
    }
});

// Security: Console Warning
if (process.env.NODE_ENV === 'production') {
    console.log('%cSTOP!', 'color: red; font-size: 50px; font-weight: bold; text-shadow: 2px 2px black;');
    console.log('%cThis is a browser feature intended for developers. If someone told you to copy-paste something here to enable a feature or "hack" someone\'s account, it is a scam and will give them access to your account.', 'font-size: 18px;');

    // Optional: Disable console.log in production
    // console.log = () => {};
    // console.warn = () => {};
    // console.error = () => {};
}

ReactDOM.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
    document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
