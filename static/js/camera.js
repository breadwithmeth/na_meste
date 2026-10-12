// страница камеры: MJPEG + overlay bounding boxes + статус.

const cameraId = window.CAMERA_ID;
const $ = (id) => document.getElementById(id);
const boxesEl = $('boxes');

const ACTION_LABELS = {
  standing: 'стоит', walking: 'идёт', sitting: 'сидит', lying: 'лежит',
  eating: 'ест/пьёт', phone: 'телефон', working: 'работает', resting: 'отдыхает',
};
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

  document.title = `${cam.name}`;
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
  let base;
  if (track.state === 'recognized' && track.employee_name) {
    const conf = track.confidence != null ? track.confidence.toFixed(2) : '';
    base = `${track.employee_name}${conf ? ' ' + conf : ''}`;
  } else if (track.state === 'unknown') {
    const conf = track.confidence != null ? track.confidence.toFixed(2) : '';
    base = `Unknown${conf ? ' ' + conf : ''}`;
  } else {
    base = '…';
  }
  // глобальная личность (межкамерный трекинг): локальный трек + global_id
  const tag = `T${track.track_id}` + (track.global_id ? `·G${track.global_id}` : '');
  return `${base} [${tag}]`;
}

function attrChipsFor(track) {
  const chips = [];
  // одежда: свотчи верх/низ
  if (track.upper_color || track.lower_color) {
    const parts = [];
    if (track.upper_color) {
      parts.push(`<span class="swatch" style="background:${track.upper_hex || '#888'}"></span>${track.upper_color}`);
    }
    if (track.lower_color) {
      parts.push(`<span class="swatch" style="background:${track.lower_hex || '#888'}"></span>${track.lower_color}`);
    }
    chips.push(`<span class="attr-chip">${parts.join('<span class="swatch-sep">·</span>')}</span>`);
  }
  // пол и возраст
  if (track.gender || track.age != null) {
    const person = [track.gender, track.age != null ? `~${track.age}` : null]
      .filter(Boolean).join(' ');
    chips.push(`<span class="attr-chip">${person}</span>`);
  }
  // предметы при человеке (нож, рюкзак, чашка…)
  for (const item of track.items || []) {
    chips.push(`<span class="attr-chip attr-item">${item}</span>`);
  }
  return chips.join('');
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

    // действие человека (сидит/работает/отдыхает/…) — чипом внутри бокса
    if (track.action && ACTION_LABELS[track.action]) {
      const chip = document.createElement('span');
      chip.className = `act-chip act-${track.action}`;
      chip.textContent = ACTION_LABELS[track.action];
      box.appendChild(chip);
    }
    // атрибуты внешности (цвет одежды, пол/возраст, предметы) — чипами
    const attrRow = document.createElement('span');
    attrRow.className = 'attr-row';
    attrRow.innerHTML = attrChipsFor(track);
    if (attrRow.innerHTML) box.appendChild(attrRow);
    boxesEl.appendChild(box);
  }

  // предметы сцены (еда на столе, посуда, оружие, носимые вещи)
  for (const obj of data.objects || []) {
    const [x1, y1, x2, y2] = obj.bbox;
    const box = document.createElement('div');
    box.className = `objbox objbox-${obj.category}`;
    box.style.left = (x1 * 100) + '%';
    box.style.top = (y1 * 100) + '%';
    box.style.width = ((x2 - x1) * 100) + '%';
    box.style.height = ((y2 - y1) * 100) + '%';

    const label = document.createElement('span');
    label.className = 'bbox-label';
    label.textContent = `${obj.label_ru} ${obj.confidence != null ? obj.confidence.toFixed(2) : ''}`;
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
