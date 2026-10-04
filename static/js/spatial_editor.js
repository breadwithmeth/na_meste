/* Редактор пространственной модели: этажи, планировка (масштаб/привязка),
стены, двери, зоны, лестницы/лифты, камеры, узлы и рёбра навигации.

Вызывается из spatial_model.js (activate/deactivate/onMapClick/...).
Мутации идут через /api/spatial/*, после каждой — App.loadWorld().
*/
(() => {
  const TOOLS = [
    { id: 'select', label: 'Выбрать', hint: 'клик по объекту — свойства' },
    { id: 'wall', label: 'Стена', hint: '2 клика: начало и конец' },
    { id: 'door', label: 'Дверь', hint: '2 клика: начало и конец' },
    { id: 'zone', label: 'Зона', hint: 'клики — вершины, двойной клик — завершить' },
    { id: 'stairs', label: 'Лестница', hint: '2 клика (линия лестницы)' },
    { id: 'elevator', label: 'Лифт', hint: '2 клика (линия лифта)' },
    { id: 'camera', label: 'Камера', hint: 'клик — позиция, параметры справа' },
    { id: 'node', label: 'Узел', hint: 'клик — позиция, тип справа' },
    { id: 'edge', label: 'Ребро', hint: 'клик по двум узлам' },
  ];

  let tool = 'select';
  let pending = [];              // точки текущей фигуры
  let edgeFrom = null;           // выбранный первый узел ребра
  let scalePicks = [];           // точки для калибровки масштаба планировки
  let scaleMode = null;          // 'scale' | 'origin' | null
  let sideEl = null;

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  // ------------------------------------------------------------- панели

  function activate(side) {
    sideEl = side;
    pending = []; edgeFrom = null; scalePicks = []; scaleMode = null;
    renderPanel();
    updateOverlay();
  }

  function deactivate() {
    window.SpatialMap.setOverlay(null);
    sideEl = null;
  }

  function renderPanel() {
    if (!sideEl) return;
    sideEl.innerHTML = '';
    sideEl.appendChild(buildTools());
    const form = el('div', 'sp-form');
    form.id = 'sp-editor-form';
    sideEl.appendChild(form);
    sideEl.appendChild(buildFloorAdmin());
    renderForm();
  }

  function buildTools() {
    const box = el('div', 'sp-tools');
    box.appendChild(el('h3', null, 'Инструменты'));
    const grid = el('div', 'sp-tool-grid');
    for (const t of TOOLS) {
      const btn = el('button', 'btn sp-tool-btn' + (tool === t.id ? ' primary' : ''), t.label);
      btn.title = t.hint;
      btn.onclick = () => {
        tool = t.id; pending = []; edgeFrom = null;
        renderPanel(); updateOverlay();
      };
      grid.appendChild(btn);
    }
    box.appendChild(grid);
    const hint = el('div', 'muted small sp-tool-hint');
    hint.id = 'sp-tool-hint';
    box.appendChild(hint);
    updateHint();
    return box;
  }

  function updateHint() {
    const hint = $('sp-tool-hint');
    const t = TOOLS.find((x) => x.id === tool);
    if (hint && t) hint.textContent = t.hint;
  }

  function buildFloorAdmin() {
    const box = el('div', 'sp-floor-admin');
    box.appendChild(el('h3', null, 'Этаж и планировка'));
    const addBtn = el('button', 'btn ghost', '+ Этаж');
    addBtn.onclick = async () => {
      const name = window.prompt('Название этажа', 'Новый этаж');
      if (!name) return;
      const z = parseFloat(window.prompt('Высота пола этажа Z, м', '0'));
      const res = await App.api('/api/spatial/floors', 'POST',
        { name, z: isNaN(z) ? 0 : z });
      if (res) await App.loadWorld(res.id);
    };
    const delBtn = el('button', 'btn danger', 'Удалить этаж');
    delBtn.onclick = async () => {
      const f = App.world.floors.find((x) => x.id === App.floorId);
      if (!f) return;
      if (!window.confirm(`Удалить «${f.name}» вместе со стенами, зонами и узлами?`)) return;
      if (await App.api(`/api/spatial/floors/${App.floorId}`, 'DELETE')) {
        await App.loadWorld();
      }
    };
    const row = el('div', 'sp-row');
    row.appendChild(addBtn); row.appendChild(delBtn);
    box.appendChild(row);

    // планировка
    const file = el('input');
    file.type = 'file';
    file.accept = 'image/*';
    file.className = 'sp-file';
    file.onchange = async () => {
      if (!file.files.length) return;
      const fd = new FormData();
      fd.append('file', file.files[0]);
      const res = await fetch(`/api/spatial/floors/${App.floorId}/floorplan`, {
        method: 'POST', body: fd,
      });
      if (!res.ok) { App.toast('Не удалось загрузить планировку'); return; }
      await App.loadWorld(App.floorId);
      App.toast('Планировка загружена. Задайте масштаб (2 клика + метры).');
    };
    box.appendChild(file);

    const scaleBtn = el('button', 'btn ghost', 'Масштаб планировки');
    scaleBtn.onclick = () => {
      if (!hasFloorplan()) return;
      scaleMode = 'scale'; scalePicks = [];
      App.toast('Кликните две точки на планировке, между которыми известно расстояние');
    };
    const originBtn = el('button', 'btn ghost', 'Привязать к миру');
    originBtn.onclick = () => {
      if (!hasFloorplan()) return;
      scaleMode = 'origin'; scalePicks = [];
      App.toast('Кликните точку на планировке с известными мировыми координатами');
    };
    const row2 = el('div', 'sp-row');
    row2.appendChild(scaleBtn); row2.appendChild(originBtn);
    box.appendChild(row2);
    const f = App.world.floors.find((x) => x.id === App.floorId);
    if (f && f.floorplan_scale) {
      box.appendChild(el('div', 'muted small',
        `Масштаб: ${(f.floorplan_scale * 100).toFixed(1)} см/пиксель` +
        (f.floorplan_origin
          ? ` · угловая точка: (${f.floorplan_origin[0]}, ${f.floorplan_origin[1]})` : '')));
    }
    return box;
  }

  function hasFloorplan() {
    const f = App.world.floors.find((x) => x.id === App.floorId);
    if (!f || !f.floorplan_scale) {
      App.toast('Сначала загрузите планировку этажа');
      return false;
    }
    return true;
  }

  // ------------------------------------------------- форма инструмента

  function renderForm() {
    const form = $('sp-editor-form');
    if (!form) return;
    form.innerHTML = '';
    const sel = App.selected;
    if (tool === 'camera') { form.appendChild(cameraForm()); return; }
    if (tool === 'node') { form.appendChild(nodeForm()); return; }
    if (tool === 'select' && sel) { form.appendChild(propsForm(sel)); return; }
    if (tool !== 'select') {
      form.appendChild(el('h3', null, TOOLS.find((t) => t.id === tool).label));
      form.appendChild(el('div', 'muted small', TOOLS.find((t) => t.id === tool).hint));
      if (pending.length) {
        form.appendChild(el('div', 'small',
          `Точек: ${pending.length}` + (tool === 'edge' && edgeFrom
            ? ` · от узла #${edgeFrom.id}` : '')));
      }
      return;
    }
    form.appendChild(el('div', 'muted small',
      'Кликните объект на карте, чтобы изменить его свойства. ' +
      'Delete — удалить выбранное.'));
  }

  function numField(label, id, value, step) {
    const wrap = el('label', 'sp-field');
    wrap.appendChild(el('span', 'sp-field-label', label));
    const input = el('input');
    input.type = 'number'; input.step = step || 'any';
    input.id = id; input.value = value != null ? value : 0;
    wrap.appendChild(input);
    return wrap;
  }

  function textField(label, id, value) {
    const wrap = el('label', 'sp-field');
    wrap.appendChild(el('span', 'sp-field-label', label));
    const input = el('input');
    input.type = 'text'; input.id = id;
    input.value = value || '';
    wrap.appendChild(input);
    return wrap;
  }

  // ---------------------------------------------------------- камера

  function cameraForm() {
    const box = el('div');
    box.appendChild(el('h3', null, 'Камера'));
    const select = el('select');
    select.id = 'sp-cam-select';
    for (const cam of App.world.cameras) {
      const opt = el('option', null, `CAM ${cam.camera_id} — ${cam.camera_name}`);
      opt.value = cam.camera_id;
      select.appendChild(opt);
    }
    if (!App.world.cameras.length) {
      box.appendChild(el('div', 'muted small', 'В системе нет камер — добавьте их на странице «Камеры»'));
      return box;
    }
    select.onchange = () => fillCameraForm();
    const wrap = el('label', 'sp-field');
    wrap.appendChild(el('span', 'sp-field-label', 'Камера'));
    wrap.appendChild(select);
    box.appendChild(wrap);
    const grid = el('div', 'sp-grid');
    grid.append(
      numField('X, м', 'sp-cam-x', 0),
      numField('Y, м', 'sp-cam-y', 0),
      numField('Высота Z, м', 'sp-cam-z', 3.0),
      numField('Yaw, °', 'sp-cam-yaw', 0),
      numField('Pitch, °', 'sp-cam-pitch', -25),
      numField('Roll, °', 'sp-cam-roll', 0),
      numField('FOV гориз., °', 'sp-cam-fov-h', 90),
      numField('FOV верт., °', 'sp-cam-fov-v', 55),
    );
    box.appendChild(grid);
    box.appendChild(el('div', 'muted small',
      'Клик по карте заполняет X/Y. Этаж — выбранный на панели сверху.'));
    const save = el('button', 'btn primary', 'Сохранить положение');
    save.onclick = async () => {
      const cameraId = parseInt(select.value, 10);
      const payload = {
        floor_id: App.floorId,
        position: { x: +$('sp-cam-x').value, y: +$('sp-cam-y').value,
                    z: +$('sp-cam-z').value },
        rotation: { yaw: +$('sp-cam-yaw').value, pitch: +$('sp-cam-pitch').value,
                    roll: +$('sp-cam-roll').value },
        fov: { horizontal: +$('sp-cam-fov-h').value,
               vertical: +$('sp-cam-fov-v').value },
        coverage_polygon: null,
      };
      if (await App.api(`/api/spatial/cameras/${cameraId}`, 'PUT', payload)) {
        App.toast('Положение камеры сохранено');
        await App.loadWorld(App.floorId);
      }
    };
    const row = el('div', 'sp-row');
    row.appendChild(save);
    box.appendChild(row);
    fillCameraForm();
    return box;
  }

  function fillCameraForm() {
    const select = $('sp-cam-select');
    if (!select) return;
    const cam = App.world.cameras.find((c) => c.camera_id === parseInt(select.value, 10));
    if (!cam || !cam.position) return;
    $('sp-cam-x').value = cam.position.x;
    $('sp-cam-y').value = cam.position.y;
    $('sp-cam-z').value = cam.position.z;
    if (cam.rotation) {
      $('sp-cam-yaw').value = cam.rotation.yaw;
      $('sp-cam-pitch').value = cam.rotation.pitch;
      $('sp-cam-roll').value = cam.rotation.roll;
    }
    if (cam.fov) {
      $('sp-cam-fov-h').value = cam.fov.horizontal;
      $('sp-cam-fov-v').value = cam.fov.vertical;
    }
  }

  // ------------------------------------------------------------ узел

  function nodeForm() {
    const box = el('div');
    box.appendChild(el('h3', null, 'Узел навигации'));
    const typeSel = el('select');
    for (const t of ['corridor', 'room', 'door', 'stairs', 'elevator',
                     'entrance', 'exit', 'restricted_zone']) {
      const opt = el('option', null, t); opt.value = t;
      typeSel.appendChild(opt);
    }
    typeSel.id = 'sp-node-type';
    const tw = el('label', 'sp-field');
    tw.appendChild(el('span', 'sp-field-label', 'Тип'));
    tw.appendChild(typeSel);
    box.appendChild(tw);
    box.appendChild(textField('Название', 'sp-node-name', ''));
    box.appendChild(numField('X, м', 'sp-node-x', 0));
    box.appendChild(numField('Y, м', 'sp-node-y', 0));
    box.appendChild(el('div', 'muted small', 'Клик по карте заполняет X/Y'));
    const add = el('button', 'btn primary', 'Добавить узел');
    add.onclick = async () => {
      const payload = {
        floor_id: App.floorId,
        type: $('sp-node-type').value,
        name: $('sp-node-name').value || null,
        position: [+$('sp-node-x').value, +$('sp-node-y').value],
      };
      if (await App.api('/api/spatial/nodes', 'POST', payload)) {
        await App.loadWorld(App.floorId);
      }
    };
    const row = el('div', 'sp-row');
    row.appendChild(add);
    box.appendChild(row);
    return box;
  }

  // ------------------------------------------------- свойства выбранного

  function propsForm(sel) {
    const box = el('div');
    box.appendChild(el('h3', null, 'Свойства'));
    if (sel.kind === 'feature') return featureProps(box, sel.obj);
    if (sel.kind === 'node') return nodeProps(box, sel.obj);
    if (sel.kind === 'camera') return cameraProps(box, sel.obj);
    return box;
  }

  function featureProps(box, feat) {
    box.appendChild(el('div', 'muted small', `Объект #${feat.id} · ${feat.type}`));
    box.appendChild(textField('Название', 'sp-prop-name', feat.name));
    const g = feat.geometry || {};
    if (g.start && g.end) {
      const grid = el('div', 'sp-grid');
      grid.append(
        numField('Start X', 'sp-prop-sx', g.start[0]),
        numField('Start Y', 'sp-prop-sy', g.start[1]),
        numField('End X', 'sp-prop-ex', g.end[0]),
        numField('End Y', 'sp-prop-ey', g.end[1]),
        numField('Высота, м', 'sp-prop-h', g.height != null ? g.height : 3.0),
      );
      box.appendChild(grid);
    } else if (g.points) {
      box.appendChild(el('div', 'muted small', `Полигон: ${g.points.length} точек`));
    }
    const save = el('button', 'btn primary', 'Сохранить');
    save.onclick = async () => {
      const payload = { name: $('sp-prop-name').value || null };
      if (g.start && g.end) {
        payload.geometry = {
          start: [+$('sp-prop-sx').value, +$('sp-prop-sy').value],
          end: [+$('sp-prop-ex').value, +$('sp-prop-ey').value],
          height: +$('sp-prop-h').value,
        };
      }
      if (await App.api(`/api/spatial/features/${feat.id}`, 'PUT', payload)) {
        await App.loadWorld(App.floorId);
      }
    };
    const del = el('button', 'btn danger', 'Удалить');
    del.onclick = async () => {
      if (await App.api(`/api/spatial/features/${feat.id}`, 'DELETE')) {
        App.select(null);
        await App.loadWorld(App.floorId);
      }
    };
    const row = el('div', 'sp-row');
    row.append(save, del);
    box.appendChild(row);
    return box;
  }

  function nodeProps(box, node) {
    box.appendChild(el('div', 'muted small', `Узел #${node.id} · ${node.type}`));
    box.appendChild(textField('Название', 'sp-prop-name', node.name));
    box.appendChild(numField('X, м', 'sp-prop-x', node.position[0]));
    box.appendChild(numField('Y, м', 'sp-prop-y', node.position[1]));
    const save = el('button', 'btn primary', 'Сохранить');
    save.onclick = async () => {
      if (await App.api(`/api/spatial/nodes/${node.id}`, 'PUT', {
        name: $('sp-prop-name').value || null,
        position: [+$('sp-prop-x').value, +$('sp-prop-y').value],
      })) await App.loadWorld(App.floorId);
    };
    const del = el('button', 'btn danger', 'Удалить');
    del.onclick = async () => {
      if (await App.api(`/api/spatial/nodes/${node.id}`, 'DELETE')) {
        App.select(null);
        await App.loadWorld(App.floorId);
      }
    };
    const row = el('div', 'sp-row');
    row.append(save, del);
    box.appendChild(row);
    return box;
  }

  function cameraProps(box, cam) {
    box.appendChild(el('div', 'muted small',
      `CAM ${cam.camera_id} — ${cam.camera_name}` +
      (cam.calibrated
        ? ` · калибровка ${(cam.reprojection_error || 0).toFixed(2)} м`
        : ' · не откалибрована')));
    const tools = el('div', 'muted small',
      'Позиция камеры редактируется инструментом «Камера»');
    box.appendChild(tools);
    return box;
  }

  // ------------------------------------------------------- клики по карте

  function onMapClick(w, e) {
    if (!sideEl) return;
    // калибровка масштаба планировки
    if (scaleMode === 'scale') {
      scalePicks.push(w);
      updateOverlay();
      if (scalePicks.length === 2) {
        const [a, b] = scalePicks;
        const px = Math.hypot(b.x - a.x, b.y - a.y)
          / App.world.floors.find((f) => f.id === App.floorId).floorplan_scale;
        const meters = parseFloat(window.prompt(
          `Расстояние между точками в пикселях планировки: ${px.toFixed(1)}\n` +
          'Введите реальное расстояние в метрах:', '5'));
        if (!isNaN(meters) && meters > 0 && px > 0) {
          const scale = meters / px;
          App.api(`/api/spatial/floors/${App.floorId}/floorplan/transform`, 'PUT', {
            scale,
            origin: App.world.floors.find((f) => f.id === App.floorId)
              .floorplan_origin || [0, 0],
          }).then(() => App.loadWorld(App.floorId));
        }
        scaleMode = null; scalePicks = []; updateOverlay();
      }
      return;
    }
    if (scaleMode === 'origin') {
      const f = App.world.floors.find((x) => x.id === App.floorId);
      const ix = (w.x - (f.floorplan_origin ? f.floorplan_origin[0] : 0))
        / f.floorplan_scale;
      const iy = (w.y - (f.floorplan_origin ? f.floorplan_origin[1] : 0))
        / f.floorplan_scale;
      const wx = parseFloat(window.prompt(
        `Пиксель планировки: (${ix.toFixed(0)}, ${iy.toFixed(0)})\n` +
        'Мировая X этой точки, м:', '0'));
      if (isNaN(wx)) { scaleMode = null; updateOverlay(); return; }
      const wy = parseFloat(window.prompt('Мировая Y этой точки, м:', '0'));
      if (isNaN(wy)) { scaleMode = null; updateOverlay(); return; }
      const origin = [wx - ix * f.floorplan_scale, wy - iy * f.floorplan_scale];
      App.api(`/api/spatial/floors/${App.floorId}/floorplan/transform`, 'PUT', {
        scale: f.floorplan_scale, origin,
      }).then(() => App.loadWorld(App.floorId));
      scaleMode = null; updateOverlay();
      return;
    }

    if (tool === 'camera') {
      const x = $('sp-cam-x'), y = $('sp-cam-y');
      if (x) x.value = w.x.toFixed(2);
      if (y) y.value = w.y.toFixed(2);
      return;
    }
    if (tool === 'node') {
      const x = $('sp-node-x'), y = $('sp-node-y');
      if (x) x.value = w.x.toFixed(2);
      if (y) y.value = w.y.toFixed(2);
      return;
    }
    if (tool === 'wall' || tool === 'door' || tool === 'stairs'
        || tool === 'elevator') {
      pending.push([w.x, w.y]);
      updateOverlay();
      if (pending.length === 2) {
        const payload = {
          type: tool,
          geometry: { start: pending[0], end: pending[1], height: 3.0 },
        };
        App.api(`/api/spatial/floors/${App.floorId}/features`, 'POST', payload)
          .then(() => App.loadWorld(App.floorId));
        pending = []; updateOverlay();
      }
      return;
    }
    if (tool === 'zone' || tool === 'restricted_zone' || tool === 'entrance'
        || tool === 'exit') {
      pending.push([w.x, w.y]);
      updateOverlay();
      return;
    }
    if (tool === 'edge') {
      const hit = e ? window.SpatialMap.hitTest(e) : null;
      if (!hit || hit.kind !== 'node') { App.toast('Ребро: кликните по узлу'); return; }
      if (!edgeFrom) { edgeFrom = hit.obj; updateOverlay(); return; }
      if (edgeFrom.id === hit.obj.id) return;
      App.api('/api/spatial/edges', 'POST', {
        from_node: edgeFrom.id, to_node: hit.obj.id,
      }).then(() => App.loadWorld(App.floorId));
      edgeFrom = null; updateOverlay();
      return;
    }
  }

  function onMapDblClick() {
    if (!sideEl) return;
    if ((tool === 'zone' || tool === 'restricted_zone' || tool === 'entrance'
         || tool === 'exit') && pending.length >= 3) {
      App.api(`/api/spatial/floors/${App.floorId}/features`, 'POST', {
        type: tool, geometry: { points: pending },
      }).then(() => App.loadWorld(App.floorId));
      pending = []; updateOverlay();
    }
  }

  function onKeydown(e) {
    if (!sideEl) return;
    if (e.key === 'Escape') {
      pending = []; edgeFrom = null; scaleMode = null; scalePicks = [];
      updateOverlay(); renderForm();
    }
    if (e.key === 'Delete' && App.selected && App.selected.kind !== 'person') {
      const sel = App.selected;
      const url = sel.kind === 'feature' ? `/api/spatial/features/${sel.obj.id}`
        : sel.kind === 'node' ? `/api/spatial/nodes/${sel.obj.id}` : null;
      if (url) {
        App.api(url, 'DELETE').then((ok) => {
          if (ok) { App.select(null); App.loadWorld(App.floorId); }
        });
      }
    }
  }

  // незавершённые фигуры поверх карты
  function updateOverlay() {
    window.SpatialMap.setOverlay((ctx, toScreen) => {
      const drawPts = (pts, color, closed) => {
        if (!pts.length) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        pts.forEach((p, i) => {
          const s = toScreen(p[0] !== undefined ? p[0] : p.x,
                             p[1] !== undefined ? p[1] : p.y, 0);
          i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y);
        });
        if (closed) ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);
        for (const p of pts) {
          const s = toScreen(p[0] !== undefined ? p[0] : p.x,
                             p[1] !== undefined ? p[1] : p.y, 0);
          ctx.beginPath();
          ctx.arc(s.x, s.y, 3, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      };
      if (pending.length) {
        drawPts(pending, '#825500',
          tool === 'zone' || tool === 'restricted_zone'
          || tool === 'entrance' || tool === 'exit');
      }
      if (scalePicks.length) drawPts(scalePicks, '#2E7D32', false);
      if (edgeFrom) {
        const s = toScreen(edgeFrom.position[0], edgeFrom.position[1], 0);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 9, 0, Math.PI * 2);
        ctx.strokeStyle = '#825500';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  }

  function refreshSelection() {
    if (sideEl && tool === 'select') renderForm();
  }

  window.SpatialEditor = {
    activate, deactivate, onMapClick, onMapDblClick, onKeydown,
    refreshSelection,
  };
})();
