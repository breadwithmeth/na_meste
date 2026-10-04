// PALEVO — глобальные личности (межкамерный трекинг): список.

const $ = (id) => document.getElementById(id);

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

async function load() {
  let persons;
  try {
    const res = await fetch('/api/global-persons?limit=200');
    if (!res.ok) throw new Error();
    persons = await res.json();
  } catch (e) {
    $('summary').textContent = 'Ошибка загрузки';
    return;
  }

  const tbody = $('persons-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('empty').hidden = persons.length > 0;
  const active = persons.filter((p) => p.status === 'ACTIVE').length;
  $('summary').textContent = persons.length
    ? `${persons.length} личностей · ${active} активных` : '';

  for (const p of persons) {
    const tr = document.createElement('tr');

    const idTd = document.createElement('td');
    const link = document.createElement('a');
    link.href = `/global/${p.global_id}`;
    link.textContent = `G#${p.global_id}`;
    idTd.appendChild(link);

    const status = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = 'status ' + (p.status === 'ACTIVE' ? 'online'
      : p.status === 'NEW' ? 'connecting' : 'offline');
    badge.textContent = p.status;
    status.appendChild(badge);

    const emp = document.createElement('td');
    emp.textContent = p.employee_name || (p.employee_id ? `#${p.employee_id}` : '—');

    const cam = document.createElement('td');
    cam.textContent = p.last_camera_name || (p.last_camera_id ? `#${p.last_camera_id}` : '—');

    const count = document.createElement('td');
    count.textContent = String(p.observations_count);

    const seen = document.createElement('td');
    seen.textContent = fmtDateTime(p.last_seen_at);

    const open = document.createElement('td');
    const openLink = document.createElement('a');
    openLink.className = 'btn';
    openLink.href = `/global/${p.global_id}`;
    openLink.textContent = 'Открыть';
    open.appendChild(openLink);

    tr.append(idTd, status, emp, cam, count, seen, open);
    tbody.appendChild(tr);
  }
}

load();
setInterval(load, 5000);
