// PALEVO — дашборд: карточки камер + статусы + счётчики людей, добавление камер.

const gridEl = document.getElementById('camera-grid');
const emptyEl = document.getElementById('empty');
const summaryEl = document.getElementById('summary');
const modal = document.getElementById('modal');
const form = document.getElementById('camera-form');
const testResult = document.getElementById('test-result');
const modalTitle = document.getElementById('modal-title');
const btnTest = document.getElementById('btn-test');

let editingId = null;

async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

async function loadCameras() {
  let cameras;
  try {
    cameras = await api('/api/cameras');
  } catch (e) {
    summaryEl.textContent = 'Ошибка загрузки: ' + e.message;
    return;
  }

  gridEl.innerHTML = '';
  emptyEl.hidden = cameras.length > 0;
  const online = cameras.filter((c) => c.status === 'ONLINE').length;
  const people = cameras.reduce((s, c) => s + (c.people_count || 0), 0);
  summaryEl.textContent = cameras.length
    ? `${cameras.length} камер · ${online} онлайн · людей в кадре: ${people}`
    : '';

  for (const cam of cameras) {
    gridEl.appendChild(renderCard(cam));
  }
}

function renderCard(cam) {
  const status = cam.status || 'OFFLINE';

  const card = document.createElement('a');
  card.className = 'cam-card';
  card.href = `/cameras/${cam.id}`;

  const head = document.createElement('div');
  head.className = 'cam-card-head';
  const name = document.createElement('span');
  name.className = 'cam-name';
  name.textContent = cam.name;
  const dot = document.createElement('span');
  dot.className = 'dot ' + status.toLowerCase();
  dot.title = status;
  head.append(name, dot);

  const meta = document.createElement('div');
  meta.className = 'cam-meta muted';
  meta.textContent = `${status} · ${cam.resolution || cam.nvr_host}`;

  const people = document.createElement('div');
  people.className = 'cam-people';
  const total = cam.people_count || 0;
  const known = cam.known_count || 0;
  const unknown = cam.unknown_count || 0;

  const badgeTotal = document.createElement('span');
  badgeTotal.className = 'p-badge';
  badgeTotal.textContent = `👥 ${total}`;
  const badgeKnown = document.createElement('span');
  badgeKnown.className = 'p-badge known';
  badgeKnown.textContent = `✓ ${known}`;
  badgeKnown.title = 'Распознанные сотрудники';
  const badgeUnknown = document.createElement('span');
  badgeUnknown.className = 'p-badge unknown';
  badgeUnknown.textContent = `? ${unknown}`;
  badgeUnknown.title = 'Неизвестные';
  people.append(badgeTotal, badgeKnown, badgeUnknown);

  const actions = document.createElement('div');
  actions.className = 'cam-card-actions';
  const edit = document.createElement('button');
  edit.className = 'btn ghost';
  edit.textContent = 'Изменить';
  edit.onclick = (e) => { e.preventDefault(); e.stopPropagation(); openEdit(cam.id); };
  const del = document.createElement('button');
  del.className = 'btn ghost danger';
  del.textContent = 'Удалить';
  del.onclick = async (e) => {
    e.preventDefault(); e.stopPropagation();
    if (!confirm(`Удалить камеру «${cam.name}»?`)) return;
    try { await api(`/api/cameras/${cam.id}`, { method: 'DELETE' }); loadCameras(); }
    catch (err) { alert('Ошибка удаления: ' + err.message); }
  };
  actions.append(edit, del);

  card.append(head, meta, people, actions);
  return card;
}

// ---------- модальная форма (как в MVP камер) ----------

function openAdd() {
  editingId = null;
  form.reset();
  form.rtsp_port.value = '554';
  form.channel.value = '1';
  form.stream_type.value = 'main';
  form.enabled.checked = true;
  form.password.required = true;
  form.password.placeholder = 'пароль NVR';
  modalTitle.textContent = 'Добавить камеру';
  hideTestResult();
  modal.hidden = false;
  form.name.focus();
}

async function openEdit(id) {
  let cam;
  try {
    cam = await api(`/api/cameras/${id}`);
  } catch (e) {
    alert('Ошибка загрузки камеры: ' + e.message);
    return;
  }
  editingId = id;
  form.reset();
  form.name.value = cam.name;
  form.nvr_host.value = cam.nvr_host;
  form.rtsp_port.value = cam.rtsp_port;
  form.username.value = cam.username;
  form.password.value = '';
  form.password.required = false;
  form.password.placeholder = '•••••••• (пусто = не менять)';
  form.channel.value = cam.channel;
  form.stream_type.value = cam.stream_type;
  form.enabled.checked = cam.enabled;
  modalTitle.textContent = `Изменить камеру #${cam.id}`;
  hideTestResult();
  modal.hidden = false;
}

function closeModal() { modal.hidden = true; }

function formPayload() {
  return {
    name: form.name.value.trim(),
    nvr_host: form.nvr_host.value.trim(),
    rtsp_port: parseInt(form.rtsp_port.value, 10),
    username: form.username.value.trim(),
    channel: parseInt(form.channel.value, 10),
    stream_type: form.stream_type.value,
    enabled: form.enabled.checked,
  };
}

function showTestResult(kind, html) {
  testResult.className = 'test-result ' + kind;
  testResult.innerHTML = html;
  testResult.hidden = false;
}

function hideTestResult() { testResult.hidden = true; }

async function testConnection() {
  const payload = formPayload();
  if (form.password.value) {
    payload.password = form.password.value;
  } else if (editingId !== null) {
    payload.camera_id = editingId; // пароль возьмём из сохранённой камеры
  } else {
    showTestResult('err', '✕ Введите пароль для проверки подключения');
    return;
  }

  btnTest.disabled = true;
  btnTest.textContent = 'Проверяю…';
  showTestResult('pending', 'Подключаюсь к RTSP…');
  try {
    const r = await api('/api/cameras/test', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (r.success) {
      showTestResult(
        'ok',
        `✓ Подключено` +
        `<div class="details">Разрешение: ${r.resolution} · Кодек: ${r.codec.toUpperCase()} · FPS: ${Math.round(r.fps || 0)}</div>`
      );
    } else {
      showTestResult('err', `✕ ${r.error || 'Не удалось подключиться к RTSP-потоку.'}`);
    }
  } catch (e) {
    showTestResult('err', '✕ Ошибка запроса: ' + e.message);
  } finally {
    btnTest.disabled = false;
    btnTest.textContent = 'Проверить подключение';
  }
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const payload = formPayload();
  try {
    if (editingId === null) {
      if (!form.password.value) { alert('Введите пароль NVR'); return; }
      payload.password = form.password.value;
      await api('/api/cameras', { method: 'POST', body: JSON.stringify(payload) });
    } else {
      if (form.password.value) payload.password = form.password.value;
      await api(`/api/cameras/${editingId}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
    }
    closeModal();
    loadCameras();
  } catch (e) {
    alert('Ошибка сохранения: ' + e.message);
  }
};

document.getElementById('btn-add').onclick = openAdd;
document.getElementById('btn-cancel').onclick = closeModal;
btnTest.onclick = testConnection;
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !modal.hidden) closeModal();
});

loadCameras();
setInterval(loadCameras, 3000);
