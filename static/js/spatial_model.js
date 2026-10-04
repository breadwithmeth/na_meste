/* Страница /spatial-model: режимы Карта / Редактор / Калибровка / Отладка.

Оркестрирует SpatialMap (canvas), SpatialEditor, SpatialCalibration.
Обновление — поллинг /api/spatial/live (1 с), как и остальные страницы
проекта (никаких вебсокетов).
*/
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  const state = {
    world: null,
    floorId: null,
    mode: 'map',
    people: [],
    selected: null,
    prediction: null,
    trail: null,
    pollCount: 0,
  };

  // ------------------------------------------------------------ helpers

  async function api(url, method, body) {
    try {
      const opts = { method: method || 'GET', headers: {} };
      if (body !== undefined) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(url, opts);
      if (res.status === 204) return true;
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        toast((data && data.detail) || `Ошибка ${res.status}`);
        return null;
      }
      return data === null ? true : data;
    } catch (err) {
      toast('Сеть недоступна');
      return null;
    }
  }

  let toastTimer = null;
  function toast(text) {
    let node = $('sp-toast');
    if (!node) {
      node = el('div', 'sp-toast');
      node.id = 'sp-toast';
      document.body.appendChild(node);
    }
    node.textContent = text;
    node.classList.add('sp-toast-show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove('sp-toast-show'), 3500);
  }

  function floorName(id) {
    const f = state.world && state.world.floors.find((x) => x.id === id);
    return f ? f.name : `этаж #${id}`;
  }

  function cameraLabel(id) {
    const cam = state.world && state.world.cameras.find((c) => c.camera_id === id);
    return cam ? `CAM ${cam.camera_id}` : `CAM ${id}`;
  }

  const COMPASS = ['В', 'ЮВ', 'Ю', 'ЮЗ', 'З', 'СЗ', 'С', 'СВ'];
  function compassName(direction) {
    if (direction == null) return '—';
    let deg = (direction * 180) / Math.PI;
    deg = ((deg % 360) + 360) % 360;
    return COMPASS[Math.round(deg / 45) % 8];
  }

  // --------------------------------------------------------- загрузка

  async function loadWorld(floorToSelect) {
    const world = await api('/api/spatial/world');
    if (!world) return;
    state.world = world;
    const select = $('sp-floor');
    select.innerHTML = '';
    for (const f of world.floors) {
      const opt = el('option', null, `${f.name} (z=${f.z})`);
      opt.value = f.id;
      select.appendChild(opt);
    }
    const target = floorToSelect != null && world.floors.some((f) => f.id === floorToSelect)
      ? floorToSelect
      : (state.floorId != null && world.floors.some((f) => f.id === state.floorId)
        ? state.floorId
        : (world.floors.length ? world.floors[0].id : null));
    state.floorId = target;
    select.value = target != null ? String(target) : '';
    window.SpatialMap.setWorld(world);
    if (target != null) window.SpatialMap.setFloor(target);
    window.SpatialMap.fit();
    $('sp-status').textContent = world.spatial_model_enabled
      ? '2.5D-модель активна'
      : '2.5D-модель выключена (SPATIAL_MODEL_ENABLED=false)';
    if (state.mode === 'editor') window.SpatialEditor.refreshSelection();
    renderSide();
  }

  // -------------------------------------------------------- селекция

  function select(sel) {
    state.selected = sel;
    state.prediction = null;
    state.trail = null;
    window.SpatialMap.setSelected(sel);
    if (state.mode === 'editor') window.SpatialEditor.refreshSelection();
    if (sel && sel.kind === 'person') {
      loadPersonDetails(sel.id);
    }
    renderSide();
  }

  async function loadPersonDetails(globalId) {
    const pred = await api(`/api/spatial/prediction/${globalId}`);
    if (state.selected && state.selected.kind === 'person'
        && state.selected.id === globalId) {
      state.prediction = pred;
      window.SpatialMap.setPrediction(pred);
      renderSide();
    }
    const traj = await api(`/api/spatial/trajectories/${globalId}?limit=600`);
    if (state.selected && state.selected.kind === 'person'
        && state.selected.id === globalId) {
      state.trail = traj ? traj.points : null;
      window.SpatialMap.setTrail(state.trail);
      renderSide();
    }
  }

  // --------------------------------------------------------- live-поллинг

  async function pollLive() {
    state.pollCount++;
    const data = await api('/api/spatial/live');
    if (!data) return;
    state.people = data.people || [];
    window.SpatialMap.setPeople(state.people);
    // выбранного человека обновляем реже (прогноз + траектория)
    if (state.selected && state.selected.kind === 'person'
        && state.pollCount % 3 === 0
        && state.people.some((p) => p.global_id === state.selected.id)) {
      loadPersonDetails(state.selected.id);
    }
    if (state.mode === 'map' || state.mode === 'debug') renderSide();
  }

  // ------------------------------------------------------ панель режимов

  function renderSide() {
    if (state.mode === 'map') renderMapSide();
    // editor/calibration строят панель сами при активации
  }

  function renderMapSide() {
    const side = $('sp-side');
    side.innerHTML = '';
    side.appendChild(el('h3', null, 'Люди сейчас'));
    if (!state.people.length) {
      side.appendChild(el('div', 'empty',
        'Никого в мировых координатах. Нужны: включённая модель ' +
        '(SPATIAL_MODEL_ENABLED), откалиброванные камеры и люди в кадре.'));
    }
    const list = el('div', 'sp-people-list');
    for (const p of state.people) {
      if (p.x == null) continue;
      const row = el('button', 'sp-person-row'
        + (state.selected && state.selected.id === p.global_id ? ' active' : ''));
      row.appendChild(el('span', 'sp-person-dot'));
      row.appendChild(el('span', null, `G#${p.global_id}`));
      row.appendChild(el('span', 'muted small',
        `${cameraLabel(p.camera_id)} · ` +
        (p.speed != null ? `${p.speed.toFixed(1)} м/с` : '')));
      row.onclick = () => select({ kind: 'person', id: p.global_id, obj: p });
      list.appendChild(row);
    }
    side.appendChild(list);
    if (state.selected && state.selected.kind === 'person') {
      side.appendChild(personDetails(state.selected));
    } else {
      side.appendChild(el('div', 'muted small',
        'Кликните человека на карте или в списке — траектория, скорость ' +
        'и предсказание.'));
    }
  }

  function personDetails(sel) {
    const box = el('div', 'sp-person-details');
    const p = state.people.find((x) => x.global_id === sel.id) || (sel.obj || {});
    box.appendChild(el('h3', null, `person_${sel.id} (G#${sel.id})`));

    const cur = el('div', 'sp-detail-grid');
    cur.appendChild(detailRow('Этаж', p.floor_id != null ? floorName(p.floor_id) : '—'));
    cur.appendChild(detailRow('X', p.x != null ? p.x.toFixed(2) : '—'));
    cur.appendChild(detailRow('Y', p.y != null ? p.y.toFixed(2) : '—'));
    cur.appendChild(detailRow('Скорость',
      p.speed != null ? `${p.speed.toFixed(2)} м/с` : '—'));
    cur.appendChild(detailRow('Направление',
      `${compassName(p.direction)} (${p.direction != null
        ? ((p.direction * 180 / Math.PI).toFixed(0) + '°') : '—'})`));
    box.appendChild(cur);

    // цепочка камер из траектории
    if (state.trail && state.trail.length) {
      const chain = [];
      for (const pt of state.trail) {
        if (pt.camera_id == null) continue;
        if (!chain.length || chain[chain.length - 1] !== pt.camera_id) {
          chain.push(pt.camera_id);
        }
      }
      if (chain.length) {
        box.appendChild(el('div', 'sp-detail-label muted small', 'Маршрут камер'));
        const traj = el('div', 'trajectory');
        chain.forEach((camId, i) => {
          if (i) traj.appendChild(el('span', 'traj-arrow', '→'));
          const node = el('span', 'traj-node', cameraLabel(camId));
          traj.appendChild(node);
        });
        box.appendChild(traj);
      }
    }
    // предсказание (ТЗ §13: «CAM 07 in ~4.2 sec»)
    if (state.prediction) {
      box.appendChild(el('div', 'sp-detail-label muted small', 'Предсказание'));
      const predBox = el('div', null);
      if (state.prediction.cameras_ahead
          && state.prediction.cameras_ahead.length) {
        for (const c of state.prediction.cameras_ahead.slice(0, 3)) {
          const line = c.eta != null
            ? `${cameraLabel(c.camera_id)} через ~${c.eta.toFixed(1)} сек`
            : `${cameraLabel(c.camera_id)} — рядом (${c.distance} м)`;
          predBox.appendChild(el('div', 'sp-predict-line', line));
        }
      } else {
        predBox.appendChild(el('div', 'muted small', 'стоит на месте'));
      }
      box.appendChild(predBox);
    } else {
      box.appendChild(el('div', 'muted small',
        'Предсказание недоступно (нет живой траектории или модель выключена)'));
    }
    const link = el('a', 'small', 'Карточка личности →');
    link.href = `/global/${sel.id}`;
    box.appendChild(link);
    return box;
  }

  function detailRow(label, value) {
    const row = el('div', 'sp-detail-row');
    row.appendChild(el('span', 'muted small', label));
    row.appendChild(el('span', null, value));
    return row;
  }

  // --------------------------------------------------------- режимы

  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll('.sp-mode-btn').forEach((b) => {
      b.classList.toggle('primary', b.dataset.mode === mode);
    });
    window.SpatialCalibration.deactivate();
    window.SpatialEditor.deactivate();
    window.SpatialMap.setDebugPoints([]);
    const hint = $('sp-hint');
    if (mode === 'editor') {
      window.SpatialEditor.activate($('sp-side'));
      hint.textContent = 'Рисуйте стены/двери кликами, Esc — отмена';
    } else if (mode === 'calibration') {
      window.SpatialCalibration.activate($('sp-side'), 'calibration');
      hint.textContent = 'Кликайте точки на кадре камеры справа';
    } else if (mode === 'debug') {
      window.SpatialCalibration.activate($('sp-side'), 'debug');
      hint.textContent = 'bbox + foot point + мировые координаты';
    } else {
      renderMapSide();
      hint.textContent = 'Колесо — зум, перетаскивание — панорама';
    }
  }

  // ------------------------------------------------------------ ввод

  function onMapClick(w, e) {
    // «взять с карты» в калибровке имеет приоритет
    if (window.SpatialCalibration.onMapClick(w)) return;
    if (state.mode === 'editor') {
      window.SpatialEditor.onMapClick(w, e);
      return;
    }
    if (state.mode !== 'map') return;
    const hit = window.SpatialMap.hitTest(e);
    if (!hit) { select(null); return; }
    if (hit.kind === 'person') {
      select({ kind: 'person', id: hit.obj.global_id, obj: hit.obj });
    } else {
      select({ kind: hit.kind, id: hit.obj.id, obj: hit.obj });
    }
  }

  function onMapDblClick(w, e) {
    if (state.mode === 'editor') window.SpatialEditor.onMapDblClick(w, e);
  }

  // -------------------------------------------------------------- init

  function init() {
    const canvas = $('sp-map');
    window.SpatialMap.init(canvas, {
      onMapClick, onMapDblClick,
    });
    document.querySelectorAll('.sp-mode-btn').forEach((b) => {
      b.onclick = () => setMode(b.dataset.mode);
    });
    $('sp-floor').onchange = (e) => {
      state.floorId = parseInt(e.target.value, 10);
      window.SpatialMap.setFloor(state.floorId);
      window.SpatialMap.fit();
      if (state.mode === 'editor') window.SpatialEditor.refreshSelection();
      renderSide();
    };
    $('sp-iso').onchange = (e) => {
      window.SpatialMap.setIso(e.target.checked);
      window.SpatialMap.fit();
    };
    $('sp-fit').onclick = () => window.SpatialMap.fit();
    document.addEventListener('keydown', (e) => {
      if (state.mode === 'editor') window.SpatialEditor.onKeydown(e);
    });

    loadWorld().then(() => {
      setMode('map');
      setInterval(pollLive, 1000);
      pollLive();
    });
  }

  // публичный API для соседних модулей
  window.App = {
    get world() { return state.world; },
    get floorId() { return state.floorId; },
    get selected() { return state.selected; },
    api, toast, loadWorld, select,
  };

  init();
})();
