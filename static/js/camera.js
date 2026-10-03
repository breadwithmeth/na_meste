// PALEVO — страница камеры: MJPEG + overlay bounding boxes + статус.

const cameraId = window.CAMERA_ID;
const $ = (id) => document.getElementById(id);
const boxesEl = $('boxes');

async function refresh() {
  let cam;
  try {
    const res = await fetch(`/api/cameras/${cameraId}`);
    if (!res.ok) throw new Error('not found');
    cam = await res.json();
  } catch (e) {
    $('video-overlay').hidden = false;
    $('video-overlay').textContent = 'Камера не найдена';
    return;
  }

  document.title = `${cam.name} — PALEVO`;
  $('camera-title').textContent = cam.name;

  const status = cam.status || 'OFFLINE';
  const badge = $('status-badge');
  badge.textContent = status;
  badge.className = 'status ' + status.toLowerCase();

  $('st-status').textContent = status;
  $('st-people').textContent = String(cam.people_count ?? 0);
  $('st-known').textContent = String(cam.known_count ?? 0);
  $('st-unknown').textContent = String(cam.unknown_count ?? 0);
  $('st-fps').textContent = cam.current_fps != null ? cam.current_fps.toFixed(1) : '—';
  $('st-res').textContent = cam.resolution || '—';
  $('st-codec').textContent = cam.codec ? cam.codec.toUpperCase() : '—';
  $('st-reconnects').textContent = cam.reconnect_count != null ? cam.reconnect_count : '—';
  $('st-lastframe').textContent = cam.seconds_since_last_frame != null
    ? `${cam.seconds_since_last_frame} сек назад` : '—';
  $('st-error').textContent = cam.error || '—';
  $('video-overlay').hidden = true;
}

// ---------- bounding boxes ----------

function labelFor(track) {
  if (track.state === 'recognized' && track.employee_name) {
    const conf = track.confidence != null ? track.confidence.toFixed(2) : '';
    return `${track.employee_name}${conf ? ' ' + conf : ''}`;
  }
  if (track.state === 'unknown') {
    const conf = track.confidence != null ? track.confidence.toFixed(2) : '';
    return `Unknown${conf ? ' ' + conf : ''}`;
  }
  return '…';
}

async function refreshBoxes() {
  let data;
  try {
    const res = await fetch(`/api/cameras/${cameraId}/detections`);
    if (!res.ok) throw new Error();
    data = await res.json();
  } catch (e) {
    return; // тихо — попробуем в следующий раз
  }

  boxesEl.innerHTML = '';
  for (const track of data.tracks || []) {
    const [x1, y1, x2, y2] = track.bbox;
    const box = document.createElement('div');
    box.className = 'bbox ' + track.state;
    box.style.left = (x1 * 100) + '%';
    box.style.top = (y1 * 100) + '%';
    box.style.width = ((x2 - x1) * 100) + '%';
    box.style.height = ((y2 - y1) * 100) + '%';

    const label = document.createElement('span');
    label.className = 'bbox-label';
    label.textContent = labelFor(track);
    box.appendChild(label);
    boxesEl.appendChild(box);
  }
}

// Если <img> не смог открыть MJPEG-поток (например, камера удалена)
$('stream').onerror = () => { $('video-overlay').hidden = false; };

refresh();
refreshBoxes();
setInterval(refresh, 2000);
setInterval(refreshBoxes, 500);
