// PALEVO — карточка глобальной личности: шапка, слияние, траектория, таймлайн.

const globalId = window.GLOBAL_ID;
const $ = (id) => document.getElementById(id);

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}
function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString('ru-RU', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

async function loadPerson() {
  let p;
  try {
    const res = await fetch(`/api/global-persons/${globalId}`);
    if (!res.ok) throw new Error();
    p = await res.json();
  } catch (e) {
    $('gp-name').textContent = 'Не найдена';
    $('gp-title').textContent = 'не найдена';
    return;
  }
  const title = p.employee_name
    ? `G#${p.global_id} · ${p.employee_name}`
    : `G#${p.global_id}`;
  document.title = `${title} — PALEVO`;
  $('gp-name').textContent = title;
  $('gp-title').textContent = `${p.status}`;

  const meta = [];
  meta.push(`Статус: <b>${p.status}</b>`);
  if (p.employee_id) {
    meta.push(`Сотрудник: <a href="/employees/${p.employee_id}">${p.employee_name || '#' + p.employee_id}</a>`);
  }
  meta.push(`Наблюдений: <b>${p.observations_count}</b>`);
  if (p.unknown_events_count > 0) {
    meta.push(`<a href="/unknown?global_id=${p.global_id}">фиксаций постороннего: ${p.unknown_events_count}</a>`);
  }
  meta.push(`Последняя камера: <b>${p.last_camera_name || (p.last_camera_id ? '#' + p.last_camera_id : '—')}</b>`);
  meta.push(`Последнее наблюдение: ${fmtDateTime(p.last_seen_at)}`);
  meta.push(`Создана: ${fmtDateTime(p.created_at)}`);
  $('gp-meta').innerHTML = meta.join(' · ');

  if (p.has_photo) {
    const img = $('gp-avatar');
    img.src = `/api/global-persons/${globalId}/photo`;
    img.hidden = false;
    img.onerror = () => { img.hidden = true; };
  }
}

async function loadTrajectory() {
  let data;
  try {
    const res = await fetch(`/api/global-persons/${globalId}/trajectory`);
    if (!res.ok) throw new Error();
    data = await res.json();
  } catch (e) { return; }

  const box = $('trajectory');
  box.innerHTML = '';
  if (!data.segments.length) {
    box.innerHTML = '<span class="muted">Траектории пока нет.</span>';
    return;
  }
  data.segments.forEach((seg, i) => {
    if (i > 0) {
      const arrow = document.createElement('span');
      arrow.className = 'traj-arrow';
      arrow.textContent = '→';
      box.appendChild(arrow);
    }
    const node = document.createElement('span');
    node.className = 'traj-node';
    node.innerHTML = `${seg.camera_name}
      <span class="muted small">${fmtTime(seg.from)}–${fmtTime(seg.to)} · ${seg.observations}</span>`;
    box.appendChild(node);
  });
}

async function loadTimeline() {
  let items;
  try {
    const res = await fetch(`/api/global-persons/${globalId}/timeline?limit=300`);
    if (!res.ok) throw new Error();
    items = await res.json();
  } catch (e) { return; }

  const tbody = $('timeline-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('timeline-empty').hidden = items.length > 0;

  for (const item of items) {
    const tr = document.createElement('tr');

    const time = document.createElement('td');
    time.textContent = fmtDateTime(item.created_at);

    const kind = document.createElement('td');
    if (item.kind === 'observation') {
      const badge = document.createElement('span');
      badge.className = 'status online';
      badge.textContent = 'наблюдение';
      kind.appendChild(badge);
    } else if (item.event_type === 'camera_transition') {
      const badge = document.createElement('span');
      badge.className = 'status reconnecting';
      badge.textContent = 'переход';
      kind.appendChild(badge);
    } else if (item.event_type === 'person_lost') {
      const badge = document.createElement('span');
      badge.className = 'status offline';
      badge.textContent = 'потерян';
      kind.appendChild(badge);
    } else {
      const badge = document.createElement('span');
      badge.className = 'status offline';
      badge.textContent = item.event_type;
      kind.appendChild(badge);
    }

    const cam = document.createElement('td');
    if (item.kind === 'event' && item.event_type === 'camera_transition') {
      const from = item.from_camera_name || `#${item.payload.from_camera}`;
      const to = item.to_camera_name || `#${item.payload.to_camera}`;
      cam.textContent = `${from} → ${to}`;
    } else {
      cam.textContent = item.camera_name || '—';
    }

    const track = document.createElement('td');
    track.textContent = item.track_id != null ? `track ${item.track_id}` : '—';

    const details = document.createElement('td');
    details.className = 'muted small';
    if (item.kind === 'event') {
      const p = item.payload || {};
      if (item.event_type === 'camera_transition') {
        details.textContent =
          `similarity=${p.similarity ?? '—'} confidence=${p.confidence ?? '—'}`;
      } else if (item.event_type === 'ambiguous_match') {
        details.textContent = `неоднозначный матч (ближе G#${p.better_global_id})`;
      } else if (item.event_type === 'person_seen') {
        details.textContent = `final=${p.final ?? '—'} reid=${p.reid ?? '—'}`
          + (p.matched === false ? ' · новая identity' : '');
      } else if (item.event_type === 'person_lost') {
        details.textContent = `длительность: ${Math.round((p.duration_seconds ?? 0) / 60)} мин`
          + (p.gallery_size ? ` · эмбеддингов: ${p.gallery_size}` : '');
      } else {
        details.textContent = JSON.stringify(p).slice(0, 80);
      }
    } else {
      details.textContent = '';
    }

    const shot = document.createElement('td');
    if (item.kind === 'observation' && item.has_snapshot) {
      const img = document.createElement('img');
      img.src = `/api/global-persons/${globalId}/observations/${item.id}/photo`;
      img.className = 'obs-thumb';
      img.loading = 'lazy';
      img.alt = 'снимок';
      shot.appendChild(img);
    } else {
      shot.textContent = '—';
    }

    tr.append(time, kind, cam, track, details, shot);
    tbody.appendChild(tr);
  }
}

// Слияние: G#source вливается в текущую личность
$('merge-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const sourceId = parseInt($('merge-source').value, 10);
  const status = $('merge-status');
  if (!sourceId) return;
  if (sourceId === globalId) {
    status.textContent = 'Это текущая личность — укажите id другой (лишней).';
    return;
  }
  status.textContent = 'Объединение…';
  try {
    const res = await fetch(`/api/global-persons/${globalId}/merge/${sourceId}`, {
      method: 'POST',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      status.textContent = data.detail || 'Ошибка объединения';
      return;
    }
    const moved = data.moved || {};
    status.textContent =
      `G#${data.merged_from} влит: наблюдений +${moved.observations ?? 0},` +
      ` событий +${moved.events ?? 0}, фиксаций +${moved.unknown_events ?? 0}` +
      (data.employee_conflict ? ' · внимание: у личностей были разные сотрудники!' : '');
    $('merge-source').value = '';
    loadPerson();
    loadTrajectory();
    loadTimeline();
  } catch (err) {
    status.textContent = 'Ошибка объединения';
  }
});

loadPerson();
loadTrajectory();
loadTimeline();
setInterval(() => { loadPerson(); loadTrajectory(); loadTimeline(); }, 5000);
