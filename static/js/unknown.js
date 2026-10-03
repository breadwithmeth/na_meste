// PALEVO — посторонние: галерея зафиксированных неопознанных людей.

const gridEl = document.getElementById('unknown-grid');
const emptyEl = document.getElementById('empty');
const summaryEl = document.getElementById('summary');

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

async function load() {
  let events;
  try {
    const res = await fetch('/api/unknown?limit=200');
    if (!res.ok) throw new Error();
    events = await res.json();
  } catch (e) {
    summaryEl.textContent = 'Ошибка загрузки';
    return;
  }

  gridEl.innerHTML = '';
  emptyEl.hidden = events.length > 0;
  summaryEl.textContent = events.length ? `зафиксировано: ${events.length}` : '';

  for (const ev of events) {
    const cell = document.createElement('div');
    cell.className = 'unknown-cell';

    const img = document.createElement('img');
    img.src = `/api/unknown/${ev.id}/photo`;
    img.alt = 'посторонний';
    img.loading = 'lazy';

    const info = document.createElement('div');
    info.className = 'unknown-info';
    const cam = document.createElement('div');
    cam.className = 'unknown-cam';
    cam.textContent = ev.camera_name || `камера #${ev.camera_id}`;
    const time = document.createElement('div');
    time.className = 'muted small';
    time.textContent = fmtDateTime(ev.created_at);
    info.append(cam, time);

    const del = document.createElement('button');
    del.className = 'btn ghost danger face-del';
    del.textContent = '✕';
    del.title = 'Удалить';
    del.onclick = async () => {
      if (!confirm('Удалить эту фиксацию?')) return;
      try {
        await fetch(`/api/unknown/${ev.id}`, { method: 'DELETE' });
        load();
      } catch (e) { alert('Ошибка удаления'); }
    };

    cell.append(img, info, del);
    gridEl.appendChild(cell);
  }
}

load();
setInterval(load, 15000);
