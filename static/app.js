// ============================================================================
// Fall Detection System - Client-side JavaScript
// ============================================================================

// Global state
let socket = null;
let statusPollInterval = null;
let isConnected = false;

// ============================================================================
// WebSocket Connection
// ============================================================================

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

        // Stop polling when connected via WebSocket
        if (statusPollInterval) {
            clearInterval(statusPollInterval);
            statusPollInterval = null;
        }
    });

    socket.on('disconnect', () => {
        console.log('[WebSocket] Disconnected');
        isConnected = false;
        updateConnectionStatus(false);

        // Fallback to polling
        startStatusPolling();
    });

    socket.on('status', (data) => {
        console.log('[WebSocket] Status update:', data);
        updateUI(data);
    });

    socket.on('alarm_triggered', (data) => {
        console.log('[WebSocket] Alarm triggered!', data);
        showAlarmBanner(data);

        // Request notification permission and show browser notification
        showBrowserNotification('ALARM!', 'Fall detected!');
    });

    socket.on('alarm_cleared', (data) => {
        console.log('[WebSocket] Alarm cleared', data);
        hideAlarmBanner();
    });
}

// ============================================================================
// Status Polling (Fallback)
// ============================================================================

function startStatusPolling() {
    if (statusPollInterval) return; // Already polling

    console.log('[Polling] Starting status polling...');

    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error('[Polling] Error:', error);
        }
    }, 1000); // Poll every 1 second
}

// ============================================================================
// UI Updates
// ============================================================================

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
    // Update fall detection status
    const fallDetectedEl = document.getElementById('fallDetected');
    if (data.fall_detected) {
        fallDetectedEl.innerHTML = '<span class="badge badge-warning">YES</span>';
    } else {
        fallDetectedEl.innerHTML = '<span class="badge badge-gray">No</span>';
    }

    // Update alarm status
    const alarmStatusEl = document.getElementById('alarmStatus');
    if (data.alarm_active) {
        alarmStatusEl.innerHTML = '<span class="badge badge-danger">ACTIVE</span>';
        showAlarmBanner(data);
    } else {
        alarmStatusEl.innerHTML = '<span class="badge badge-gray">Inactive</span>';
        hideAlarmBanner();
    }

    // Update fall duration
    const fallDurationEl = document.getElementById('fallDuration');
    fallDurationEl.innerHTML = `<span class="duration">${data.fall_duration}s</span>`;

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

// ============================================================================
// Browser Notifications
// ============================================================================

function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

function showBrowserNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        const notification = new Notification(title, {
            body: body,
            icon: '/static/favicon.ico', // Optional: add your icon
            tag: 'fall-detection-alarm',
            requireInteraction: true
        });

        notification.onclick = () => {
            window.focus();
            notification.close();
        };

        // Auto-close after 10 seconds
        setTimeout(() => notification.close(), 10000);
    }
}

// ============================================================================
// Video Feed
// ============================================================================

function initializeVideoFeed() {
    const videoFeed = document.getElementById('videoFeed');
    const videoOverlay = document.getElementById('videoOverlay');

    // Hide overlay when video loads
    videoFeed.addEventListener('load', () => {
        videoOverlay.classList.add('hidden');
    });

    // Show overlay on error
    videoFeed.addEventListener('error', () => {
        videoOverlay.classList.remove('hidden');
        videoOverlay.querySelector('p').textContent = 'Camera load error';
    });
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('[App] Initializing...');

    // Initialize video feed
    initializeVideoFeed();

    // Request notification permission
    requestNotificationPermission();

    // Initialize WebSocket
    initializeWebSocket();

    // Start polling as fallback (will be stopped if WebSocket connects)
    startStatusPolling();

    console.log('[App] Initialization complete');
});

// ============================================================================
// Cleanup
// ============================================================================

window.addEventListener('beforeunload', () => {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
    }
    if (socket) {
        socket.disconnect();
    }
});
