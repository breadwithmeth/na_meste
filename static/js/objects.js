// предметы сцены (еда/посуда/оружие/носимые) + внешность людей.

const $ = (id) => document.getElementById(id);

const CATEGORY_LABELS = {
  weapon: 'оружие', food: 'еда', tableware: 'посуда',
  carried: 'носимые вещи', furniture: 'мебель', other: 'прочее',
};

function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString('ru-RU', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  });
}

function categoryChip(category, label) {
  return `<span class="obj-chip obj-${category}">${label || CATEGORY_LABELS[category] || category}</span>`;
}

function swatch(color, hex) {
  if (!color) return '—';
  return `<span class="swatch-row"><span class="swatch" style="background:${hex || '#888'}"></span>${color}</span>`;
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
    const res = await fetch('/api/objects/live');
    items = await res.json();
  } catch (e) { return; }

  const tbody = $('live-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('live-empty').hidden = items.length > 0;
  $('live-summary').textContent = items.length
    ? `${items.length} предмет(ов) в кадре` : '';

  for (const o of items) {
    tbody.appendChild(row([
      o.label_ru || o.label,
      categoryChip(o.category, o.category_label),
      o.camera_name || `#${o.camera_id}`,
      o.confidence != null ? o.confidence.toFixed(2) : '—',
      o.track_ids && o.track_ids.length ? `рядом с треком T${o.track_ids.join(', T')}` : '—',
    ]));
  }
}

// ------------------------------------------------------------- история

let historyCategory = '';

async function loadHistory() {
  let items;
  try {
    const res = await fetch(`/api/objects?limit=200${historyCategory ? `&category=${historyCategory}` : ''}`);
    items = await res.json();
  } catch (e) { return; }

  const tbody = $('history-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('history-empty').hidden = items.length > 0;

  for (const it of items) {
    const who = it.employee_name
      || (it.employee_id ? `сотрудник #${it.employee_id}`
        : (it.global_id ? `G#${it.global_id}` : '—'));
    tbody.appendChild(row([
      fmtDateTime(it.created_at),
      (it.label_ru || it.label) + (it.confidence != null ? ` · ${it.confidence.toFixed(2)}` : ''),
      categoryChip(it.category, it.category_label),
      it.camera_name || `#${it.camera_id}`,
      who,
      it.has_snapshot ? `<a class="muted" href="/api/objects/${it.id}/snapshot" target="_blank">снимок</a>` : '—',
    ]));
  }
}

// ------------------------------------------------------------- внешность

async function loadAppearance() {
  let items;
  try {
    const res = await fetch('/api/appearance/latest');
    items = await res.json();
  } catch (e) { return; }

  const tbody = $('appearance-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('appearance-empty').hidden = items.length > 0;

  for (const it of items) {
    const who = it.employee_name
      || (it.global_id ? `Посторонний G#${it.global_id}` : 'Неизвестный');
    const person = [it.gender, it.age != null ? `~${it.age}` : null]
      .filter(Boolean).join(' ') || '—';
    tbody.appendChild(row([
      who,
      it.camera_name || `#${it.camera_id}`,
      swatch(it.upper_color, it.upper_hex),
      swatch(it.lower_color, it.lower_hex),
      person,
      fmtDateTime(it.created_at),
    ]));
  }
}

$('category-filter').addEventListener('change', (e) => {
  historyCategory = e.target.value;
  loadHistory();
});

loadLive();
loadHistory();
loadAppearance();
setInterval(loadLive, 2000);
setInterval(loadHistory, 30000);
setInterval(loadAppearance, 30000);
