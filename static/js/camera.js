// Camera Monitor — страница камеры: живое превью + панель состояния.

const cameraId = window.CAMERA_ID;
const $ = (id) => document.getElementById(id);

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

  document.title = `${cam.name} — Camera Monitor`;
  $('camera-title').textContent = cam.name;

  const status = cam.status || 'OFFLINE';
  const badge = $('status-badge');
  badge.textContent = status;
  badge.className = 'status ' + status.toLowerCase();

  $('st-status').textContent = status;
  $('st-fps').textContent =
    cam.current_fps != null ? cam.current_fps.toFixed(1) : '—';
  $('st-res').textContent = cam.resolution || '—';
  $('st-codec').textContent = cam.codec ? cam.codec.toUpperCase() : '—';
  $('st-reconnects').textContent =
    cam.reconnect_count != null ? cam.reconnect_count : '—';
  $('st-lastframe').textContent =
    cam.seconds_since_last_frame != null
      ? `${cam.seconds_since_last_frame} сек назад`
      : '—';
  $('st-error').textContent = cam.error || '—';
  $('video-overlay').hidden = true;
}

// Если <img> не смог открыть MJPEG-поток (например, камера удалена)
$('stream').onerror = () => {
  $('video-overlay').hidden = false;
};

refresh();
setInterval(refresh, 2000);
