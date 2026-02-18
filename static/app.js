// =============================================================================
// Fall Detection System - Client-side JavaScript
// =============================================================================

let socket = null;
let statusPollInterval = null;
let isConnected = false;

// Zone Editor State
let isEditingZones = false;
let currentZoneType = 'bed'; // 'bed' or 'door'
let isDrawing = false;
let startX, startY;
let zones = { bed: [], door: [] };
let canvas, ctx;

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
    } else if (state === 'RESTING') {
        fallStateEl.innerHTML = '<span class="badge badge-primary">RESTING</span>';
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
// Zone Editor (Polygon Support)
// =============================================================================

function initializeZoneEditor() {
    canvas = document.getElementById('zoneCanvas');
    ctx = canvas.getContext('2d');
    const videoFeed = document.getElementById('videoFeed');

    // Sync canvas size to video size
    canvas.width = videoFeed.clientWidth || 640;
    canvas.height = videoFeed.clientHeight || 480;

    window.addEventListener('resize', () => {
        canvas.width = videoFeed.clientWidth || 640;
        canvas.height = videoFeed.clientHeight || 480;
        if (isEditingZones) drawZones();
    });

    // Polygon drawing events
    canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('mousemove', handleCanvasMove);
    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        finishPolygon();
    });
}

function toggleZoneEditor() {
    const controls = document.getElementById('zoneEditorControls');
    isEditingZones = !isEditingZones;

    if (isEditingZones) {
        controls.classList.remove('hidden');
        canvas.style.pointerEvents = 'auto'; // Enable interaction
        loadZones(); // Load current zones from backend
        currentPolygon = [];
    } else {
        controls.classList.add('hidden');
        canvas.style.pointerEvents = 'none'; // Passthrough
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear overlay
        currentPolygon = [];
    }
}

function setZoneType(type) {
    currentZoneType = type;
    const btns = document.querySelectorAll('.zone-type-selector button');
    btns.forEach(btn => btn.classList.remove('active'));

    // Find button by text content or logic
    if (type === 'bed') btns[0].classList.add('active');
    if (type === 'door') btns[1].classList.add('active');
}

async function loadZones() {
    try {
        const response = await fetch('/api/zones');
        zones = await response.json();
        drawZones();
    } catch (e) {
        console.error('Failed to load zones', e);
    }
}

async function saveZones() {
    try {
        await fetch('/api/zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(zones)
        });
        toggleZoneEditor(); // Close editor
    } catch (e) {
        console.error('Failed to save zones', e);
        alert('Failed to save zones');
    }
}

function clearZones() {
    zones = { bed: [], door: [] };
    currentPolygon = [];
    drawZones();
}

function handleCanvasClick(e) {
    if (!isEditingZones) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / canvas.width;
    const y = (e.clientY - rect.top) / canvas.height;

    // Add point to current polygon
    currentPolygon.push([x, y]);
    drawZones();
}

function handleCanvasMove(e) {
    if (!isEditingZones || currentPolygon.length === 0) return;
    drawZones();

    // Draw guide line from last point to mouse cursor
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const lastPt = currentPolygon[currentPolygon.length - 1];

    ctx.beginPath();
    ctx.moveTo(lastPt[0] * canvas.width, lastPt[1] * canvas.height);
    ctx.lineTo(mouseX, mouseY);
    ctx.strokeStyle = currentZoneType === 'bed' ? '#007bff' : '#28a745';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
}

function finishPolygon() {
    if (currentPolygon.length >= 3) {
        zones[currentZoneType].push([...currentPolygon]);
    }
    currentPolygon = [];
    drawZones();
}

function drawZones() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Helper to draw a single polygon
    const drawPoly = (points, color, fillColor, label) => {
        if (points.length < 1) return;

        ctx.beginPath();
        ctx.moveTo(points[0][0] * canvas.width, points[0][1] * canvas.height);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0] * canvas.width, points[i][1] * canvas.height);
        }
        ctx.closePath();

        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        if (label) {
            ctx.fillStyle = color;
            ctx.font = '12px Arial';
            ctx.fillText(label, points[0][0] * canvas.width, points[0][1] * canvas.height - 5);
        }
    };

    // Draw saved Bed Zones
    zones.bed.forEach(poly => drawPoly(poly, '#007bff', 'rgba(0, 123, 255, 0.2)', 'BED'));

    // Draw saved Door Zones
    zones.door.forEach(poly => drawPoly(poly, '#28a745', 'rgba(40, 167, 69, 0.2)', 'DOOR'));

    // Draw current polygon in progress
    if (currentPolygon.length > 0) {
        const color = currentZoneType === 'bed' ? '#007bff' : '#28a745';

        // Draw points
        ctx.fillStyle = color;
        currentPolygon.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0] * canvas.width, pt[1] * canvas.height, 4, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw lines between points
        if (currentPolygon.length > 1) {
            ctx.beginPath();
            ctx.moveTo(currentPolygon[0][0] * canvas.width, currentPolygon[0][1] * canvas.height);
            for (let i = 1; i < currentPolygon.length; i++) {
                ctx.lineTo(currentPolygon[i][0] * canvas.width, currentPolygon[i][1] * canvas.height);
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
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
    initializeZoneEditor(); // Initialize canvas
    requestNotificationPermission();
    initializeWebSocket();
    startStatusPolling();

    // Make wrapper functions global for onclick handlers
    window.toggleZoneEditor = toggleZoneEditor;
    window.setZoneType = setZoneType;
    window.saveZones = saveZones;
    window.clearZones = clearZones;
});

window.addEventListener('beforeunload', () => {
    if (statusPollInterval) clearInterval(statusPollInterval);
    if (socket) socket.disconnect();
});
