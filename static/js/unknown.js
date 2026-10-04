// посторонние: галерея зафиксированных неопознанных людей.

const gridEl = document.getElementById('unknown-grid');
const emptyEl = document.getElementById('empty');
const summaryEl = document.getElementById('summary');

// ?global_id=184 — показать фиксации одной глобальной личности
// (переход со страницы человека и обратно)
const filterGid = new URLSearchParams(window.location.search).get('global_id');

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

async function load() {
  let events;
  try {
    const url = filterGid
      ? `/api/unknown?limit=500&global_id=${encodeURIComponent(filterGid)}`
      : '/api/unknown?limit=200';
    const res = await fetch(url);
    if (!res.ok) throw new Error();
    events = await res.json();
  } catch (e) {
    summaryEl.textContent = 'Ошибка загрузки';
    return;
  }

  gridEl.innerHTML = '';
  emptyEl.hidden = events.length > 0;
  if (filterGid) {
    const back = document.createElement('a');
    back.href = `/global/${filterGid}`;
    back.textContent = `← G#${filterGid}`;
    back.className = 'gid-badge';
    back.title = 'Вернуться к личности';
    summaryEl.innerHTML = '';
    summaryEl.append(`фиксации G#${filterGid}: ${events.length} · `, back,
      ' · ', Object.assign(document.createElement('a'),
        { href: '/unknown', textContent: 'показать всех' }));
  } else {
    summaryEl.textContent = events.length ? `зафиксировано: ${events.length}` : '';
  }

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
    // трекинг постороннего: ссылка на глобальную личность + сколько фиксаций
    if (ev.global_id) {
      const gid = document.createElement('a');
      gid.className = 'gid-badge';
      gid.href = `/global/${ev.global_id}`;
      const samePerson = events.filter((e) => e.global_id === ev.global_id).length;
      gid.textContent = `G#${ev.global_id}`
        + (samePerson > 1 ? ` · фиксаций: ${samePerson}` : '');
      gid.title = 'Глобальная личность — таймлайн и траектория';
      cam.appendChild(document.createTextNode(' '));
      cam.appendChild(gid);
    }
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
