// =============================================================================
// Fall Detection System - Client-side JavaScript
// =============================================================================

let socket = null;
let statusPollInterval = null;
let isConnected = false;

// =============================================================================
// WebSocket Connection
// =============================================================================

function initializeWebSocket() {
    console.log('[WebSocket] Connecting...');

    socket = io({
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: Infinity
    });

    socket.on('connect', () => {
        console.log('[WebSocket] Connected');
        isConnected = true;
        updateConnectionStatus(true);

        if (statusPollInterval) {
            clearInterval(statusPollInterval);
            statusPollInterval = null;
        }
    });

    socket.on('disconnect', () => {
        console.log('[WebSocket] Disconnected');
        isConnected = false;
        updateConnectionStatus(false);
        startStatusPolling();
    });

    socket.on('status', (data) => {
        updateUI(data);
    });

    socket.on('alarm_triggered', (data) => {
        console.log('[WebSocket] Alarm triggered!', data);
        showAlarmBanner(data);
        showBrowserNotification('ALARM!', 'Fall detected!');
    });

    socket.on('alarm_cleared', (data) => {
        console.log('[WebSocket] Alarm cleared', data);
        hideAlarmBanner();
    });
}

// =============================================================================
// Status Polling (Fallback)
// =============================================================================

function startStatusPolling() {
    if (statusPollInterval) return;

    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error('[Polling] Error:', error);
        }
    }, 1000);
}

// =============================================================================
// UI Updates
// =============================================================================

function updateConnectionStatus(connected) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');

    if (connected) {
        statusDot.classList.add('connected');
        statusDot.classList.remove('disconnected');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.remove('connected');
        statusDot.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}

function updateUI(data) {
    // Update fall state
    const fallStateEl = document.getElementById('fallState');
    const state = data.fall_state || 'MONITORING';

    if (state === 'ALARM') {
        fallStateEl.innerHTML = '<span class="badge badge-danger">ALARM</span>';
        showAlarmBanner(data);
    } else if (state === 'ALERT') {
        fallStateEl.innerHTML = '<span class="badge badge-warning">ALERT</span>';
    } else if (state === 'CAUTION') {
        fallStateEl.innerHTML = '<span class="badge badge-warning">CAUTION</span>';
    } else {
        fallStateEl.innerHTML = '<span class="badge badge-success">MONITORING</span>';
        hideAlarmBanner();
    }

    // Update fall score
    const fallScoreEl = document.getElementById('fallScore');
    const score = data.fall_score || 0;
    fallScoreEl.innerHTML = `<span class="duration">${score.toFixed(2)}</span>`;

    // Update alarm status
    const alarmStatusEl = document.getElementById('alarmStatus');
    if (data.alarm_active) {
        alarmStatusEl.innerHTML = '<span class="badge badge-danger">ACTIVE</span>';
    } else {
        alarmStatusEl.innerHTML = '<span class="badge badge-gray">Inactive</span>';
    }

    // Update last update time
    const lastUpdateEl = document.getElementById('lastUpdate');
    const timestamp = new Date(data.timestamp);
    const timeString = timestamp.toLocaleTimeString('en-US');
    lastUpdateEl.innerHTML = `<span class="time">${timeString}</span>`;
}

function showAlarmBanner(data) {
    const banner = document.getElementById('alarmBanner');
    const alarmTime = document.getElementById('alarmTime');

    const timestamp = new Date(data.timestamp);
    const timeString = timestamp.toLocaleTimeString('en-US');
    alarmTime.textContent = `Detected at ${timeString}`;

    banner.classList.remove('hidden');
}

function hideAlarmBanner() {
    const banner = document.getElementById('alarmBanner');
    banner.classList.add('hidden');
}

// =============================================================================
// Browser Notifications
// =============================================================================

function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

function showBrowserNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        const notification = new Notification(title, {
            body: body,
            tag: 'fall-detection-alarm',
            requireInteraction: true
        });

        notification.onclick = () => {
            window.focus();
            notification.close();
        };

        setTimeout(() => notification.close(), 10000);
    }
}

// =============================================================================
// Video Feed
// =============================================================================

function initializeVideoFeed() {
    const videoFeed = document.getElementById('videoFeed');
    const videoOverlay = document.getElementById('videoOverlay');

    videoFeed.addEventListener('load', () => {
        videoOverlay.classList.add('hidden');
    });

    videoFeed.addEventListener('error', () => {
        videoOverlay.classList.remove('hidden');
        videoOverlay.querySelector('p').textContent = 'Camera load error';
    });
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeVideoFeed();
    requestNotificationPermission();
    initializeWebSocket();
    startStatusPolling();
});

window.addEventListener('beforeunload', () => {
    if (statusPollInterval) clearInterval(statusPollInterval);
    if (socket) socket.disconnect();
});
