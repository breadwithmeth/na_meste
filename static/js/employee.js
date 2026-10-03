// PALEVO — карточка сотрудника: данные, загрузка фото лиц, история присутствия.

const employeeId = window.EMPLOYEE_ID;
const $ = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const res = await fetch(url, { ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// ---------- данные сотрудника ----------

async function loadEmployee() {
  let emp;
  try {
    emp = await api(`/api/employees/${employeeId}`);
  } catch (e) {
    alert('Сотрудник не найден');
    location.href = '/employees';
    return;
  }
  document.title = `${emp.name} — PALEVO`;
  $('emp-title').textContent = emp.name;
  const f = $('emp-form');
  f.name.value = emp.name;
  f.external_id.value = emp.external_id || '';
  f.active.checked = emp.active;
  $('faces-count').textContent = emp.faces_count ? `· ${emp.faces_count}` : '';
}

$('emp-form').onsubmit = async (e) => {
  e.preventDefault();
  const f = $('emp-form');
  try {
    await api(`/api/employees/${employeeId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: f.name.value.trim(),
        external_id: f.external_id.value.trim() || null,
        active: f.active.checked,
      }),
    });
    loadEmployee();
  } catch (err) {
    alert('Ошибка сохранения: ' + err.message);
  }
};

// ---------- загрузка фото ----------

$('btn-upload').onclick = () => $('file-input').click();
$('file-input').onchange = () => uploadFiles($('file-input').files);

async function uploadFiles(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;

  for (const file of files) {
    const li = addStatusItem(file.name);
    setStep(li, 'pending', 'Поиск лица…');
    const fd = new FormData();
    fd.append('files', file);
    try {
      const res = await fetch(`/api/employees/${employeeId}/faces`, {
        method: 'POST',
        body: fd,
      });
      const data = await res.json();
      const r = (data.results || [])[0];
      if (!res.ok || !r || !r.success) {
        const msg = (r && r.error) || data.detail || 'Ошибка обработки';
        setStep(li, 'err', `✕ ${msg}`);
        continue;
      }
      let text = '✓ Лицо найдено · Эмбеддинг создан · Сохранено';
      if (r.warnings && r.warnings.length) {
        text += ` <div class="warn">⚠ ${r.warnings.join('; ')}</div>`;
      }
      setStep(li, 'ok', text);
    } catch (e) {
      setStep(li, 'err', '✕ Ошибка загрузки: ' + e.message);
    }
  }
  $('file-input').value = '';
  loadFaces();
  loadEmployee();
}

function addStatusItem(filename) {
  const li = document.createElement('li');
  li.className = 'upload-item';
  li.innerHTML = `<span class="upload-name"></span><span class="upload-step"></span>`;
  li.querySelector('.upload-name').textContent = filename;
  $('upload-status').prepend(li);
  return li;
}

function setStep(li, kind, html) {
  li.className = `upload-item ${kind}`;
  li.querySelector('.upload-step').innerHTML = html;
}

// ---------- список лиц ----------

async function loadFaces() {
  let emp;
  try {
    emp = await api(`/api/employees/${employeeId}?faces=1`);
  } catch (e) { return; }
  const grid = $('faces-grid');
  grid.innerHTML = '';
  const faces = emp.faces || [];
  $('faces-count').textContent = faces.length ? `· ${faces.length}` : '';
  for (const face of faces) {
    const cell = document.createElement('div');
    cell.className = 'face-cell';
    const img = document.createElement('img');
    img.src = `/api/employees/${employeeId}/faces/${face.id}/thumbnail`;
    img.alt = 'лицо';
    const del = document.createElement('button');
    del.className = 'btn ghost danger face-del';
    del.textContent = '✕';
    del.title = 'Удалить лицо';
    del.onclick = async () => {
      if (!confirm('Удалить это лицо?')) return;
      try {
        await api(`/api/employees/${employeeId}/faces/${face.id}`, { method: 'DELETE' });
        loadFaces();
        loadEmployee();
      } catch (e) { alert('Ошибка удаления: ' + e.message); }
    };
    cell.append(img, del);
    grid.appendChild(cell);
  }
}

// ---------- история присутствия ----------

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function loadPresence() {
  let rows;
  try {
    rows = await api(`/api/presence/employee/${employeeId}`);
  } catch (e) { return; }

  const tbody = $('presence-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('presence-empty').hidden = rows.length > 0;
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.camera_name || '—'}</td>
      <td>${fmtTime(r.started_at)}</td>
      <td>${fmtTime(r.last_seen_at)}</td>
      <td>${r.ended_at ? fmtTime(r.ended_at) : '<span class="status online">открыта</span>'}</td>
      <td>${r.confidence != null ? r.confidence.toFixed(2) : '—'}</td>`;
    tbody.appendChild(tr);
  }
}

loadEmployee();
loadFaces();
loadPresence();
