// действия людей: сейчас / сводка / история.

const $ = (id) => document.getElementById(id);

const ACTION_LABELS = {
  standing: 'стоит', walking: 'идёт', sitting: 'сидит', lying: 'лежит',
  eating: 'ест/пьёт', phone: 'телефон', working: 'работает', resting: 'отдыхает',
};

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
  return fmtSeconds((to - from) / 1000);
}

function fmtSeconds(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec} сек`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m} мин`;
  return `${Math.floor(m / 60)} ч ${m % 60} мин`;
}

function actionChip(action, label) {
  return `<span class="act-chip act-${action}">${label || ACTION_LABELS[action] || action}</span>`;
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

// ------------------------------------------------------------- сейчас

async function loadLive() {
  let items;
  try {
    const res = await fetch('/api/actions/live');
    items = await res.json();
  } catch (e) { return; }

  const tbody = $('live-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('live-empty').hidden = items.length > 0;
  $('live-summary').textContent = items.length
    ? `${items.length} человек(а) в кадре` : '';

  for (const t of items) {
    const who = t.employee_name
      || (t.global_id ? `Посторонний G#${t.global_id}` : 'Неизвестный');
    tbody.appendChild(row([
      who,
      t.camera_name || `#${t.camera_id}`,
      actionChip(t.action, t.action_label),
      t.action_confidence != null ? t.action_confidence.toFixed(2) : '—',
      fmtTime(t.action_since),
      fmtDuration(t.action_since, null),
    ]));
  }
}

// ------------------------------------------------------------- сводка

async function loadSummary() {
  let persons;
  try {
    const hours = encodeURIComponent($('summary-hours').value || '8');
    const res = await fetch(`/api/actions/summary?hours=${hours}`);
    persons = await res.json();
  } catch (e) { return; }

  const tbody = $('summary-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('summary-empty').hidden = persons.length > 0;

  for (const p of persons) {
    const who = p.employee_name
      || (p.global_id ? `Посторонний G#${p.global_id}` : 'Неизвестный');
    const chips = Object.entries(p.actions)
      .map(([a, sec]) => `${actionChip(a)} ${fmtSeconds(sec)}`)
      .join(' · ');
    tbody.appendChild(row([
      who,
      chips || '—',
      fmtSeconds(p.total_seconds),
    ]));
  }
}

// ------------------------------------------------------------- история

let historyAction = '';

async function loadHistory() {
  let items;
  try {
    const res = await fetch(`/api/actions?limit=200${historyAction ? `&action=${historyAction}` : ''}`);
    items = await res.json();
  } catch (e) { return; }

  const tbody = $('history-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('history-empty').hidden = items.length > 0;

  for (const it of items) {
    const who = it.employee_name
      || (it.global_id ? `G#${it.global_id}` : '—');
    tbody.appendChild(row([
      fmtTime(it.created_at),
      who,
      it.camera_name || `#${it.camera_id}`,
      actionChip(it.action, it.action_label),
      it.confidence != null ? it.confidence.toFixed(2) : '—',
      `T${it.track_id}`,
    ]));
  }
}

$('summary-hours').addEventListener('change', loadSummary);
$('history-filter').addEventListener('change', (e) => {
  historyAction = e.target.value;
  loadHistory();
});

loadLive();
loadSummary();
loadHistory();
setInterval(loadLive, 2000);
setInterval(loadSummary, 30000);
setInterval(loadHistory, 30000);
