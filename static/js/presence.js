// PALEVO — присутствие: активные сессии + история.

const $ = (id) => document.getElementById(id);

function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString('ru-RU', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function fmtDuration(fromIso, toIso) {
  if (!fromIso) return '—';
  const from = new Date(fromIso).getTime();
  const to = toIso ? new Date(toIso).getTime() : Date.now();
  const sec = Math.max(0, Math.round((to - from) / 1000));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m} мин ${s} сек` : `${s} сек`;
}

function row(tds) {
  const tr = document.createElement('tr');
  for (const td of tds) {
    const el = document.createElement('td');
    if (td instanceof HTMLElement) el.appendChild(td);
    else el.innerHTML = td;
    tr.appendChild(el);
  }
  return tr;
}

async function loadActive() {
  let sessions;
  try {
    const res = await fetch('/api/presence/active');
    sessions = await res.json();
  } catch (e) { return; }

  const tbody = $('active-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('active-empty').hidden = sessions.length > 0;
  $('active-summary').textContent = sessions.length
    ? `${sessions.length} сейчас на камерах` : '';

  for (const s of sessions) {
    const open = document.createElement('span');
    open.className = 'status online';
    open.textContent = 'здесь';
    tbody.appendChild(row([
      s.employee_name || `#${s.employee_id}`,
      s.camera_name || `#${s.camera_id}`,
      fmtTime(s.started_at),
      `${fmtTime(s.last_seen_at)} (${fmtDuration(s.started_at, null)})`,
      s.confidence != null ? s.confidence.toFixed(2) : '—',
    ]));
  }
}

async function loadHistory() {
  let sessions;
  try {
    const res = await fetch('/api/presence?limit=200');
    sessions = await res.json();
  } catch (e) { return; }

  const tbody = $('history-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('history-empty').hidden = sessions.length > 0;

  for (const s of sessions) {
    tbody.appendChild(row([
      s.employee_name || `#${s.employee_id}`,
      s.camera_name || `#${s.camera_id}`,
      fmtTime(s.started_at),
      fmtTime(s.last_seen_at),
      s.ended_at ? fmtTime(s.ended_at) : '<span class="status online">открыта</span>',
      s.confidence != null ? s.confidence.toFixed(2) : '—',
    ]));
  }
}

loadActive();
loadHistory();
setInterval(loadActive, 5000);
setInterval(loadHistory, 30000);
