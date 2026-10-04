/* Калибровка камер и Calibration debug (ТЗ §4, §15).

Калибровка: снимок камеры → ≥4 кликов по кадру (pixel) + мировые
координаты (вручную или кликом по карте) → POST /calibration →
гомография и зона видимости на сервере, здесь — сетка-проверка.

Отладка: live MJPEG + bbox + foot point + мировые координаты (через
гомографию в JS); те же точки одновременно показываются на карте.
*/
(() => {
  let sideEl = null;
  let mode = null;                // 'calibration' | 'debug'
  let points = [];                // [{pixel:[x,y], world:[x,y]|null}]
  let pickIndex = null;           // индекс точки, ждущей клика по карте
  let snapshot = null;            // {img, canvas, ctx, w, h}
  let live = null;                // {img, canvas, ctx, timer, cameraId}
  let calibration = null;         // последняя калибровка камеры (из GET)
  let gridOn = false;
  // маркеры: физические объекты — world-координаты общие для всех камер,
  // меняется только факт видимости в кадре текущей камеры
  let markerRows = [];            // [{id, world:[x,y]|null, detected, sizePx}]
  let detectedMarkers = [];       // [{id, pixel:[x,y], size}] для overlay

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  function activate(side, m) {
    sideEl = side;
    mode = m;
    points = []; pickIndex = null; calibration = null; gridOn = false;
    renderPanel();
  }

  function deactivate() {
    if (live && live.timer) clearInterval(live.timer);
    live = null; snapshot = null; sideEl = null;
    window.SpatialMap.setDebugPoints([]);
  }

  function cameraSelect(id, onchange) {
    const select = el('select');
    for (const cam of App.world.cameras) {
      const opt = el('option', null,
        `CAM ${cam.camera_id} — ${cam.camera_name}` +
        (cam.calibrated ? ' ✓' : ''));
      opt.value = cam.camera_id;
      select.appendChild(opt);
    }
    select.id = id;
    select.onchange = onchange;
    return select;
  }

  function cameraIds() {
    return App.world.cameras.map((c) => c.camera_id);
  }

  // ---------------------------------------------------------- калибровка

  function renderPanel() {
    if (!sideEl) return;
    sideEl.innerHTML = '';
    if (!cameraIds().length) {
      sideEl.appendChild(el('div', 'empty',
        'В системе нет камер — добавьте их на странице «Камеры».'));
      return;
    }
    if (mode === 'calibration') renderCalibration();
    else renderDebug();
  }

  function renderCalibration() {
    sideEl.appendChild(el('h3', null, 'Калибровка камеры'));
    const wrap = el('label', 'sp-field');
    wrap.appendChild(el('span', 'sp-field-label', 'Камера'));
    wrap.appendChild(cameraSelect('cal-camera', onCameraChange));
    sideEl.appendChild(wrap);
    sideEl.appendChild(el('div', 'muted small',
      'Отметьте минимум 4 известных точки НА ПОЛУ: клик по кадру — pixel, ' +
      'мировые координаты вручную или кнопкой «карта».'));

    // кадр + overlay
    const frameWrap = el('div', 'sp-frame-wrap');
    const img = el('img', 'sp-frame');
    img.id = 'cal-img';
    img.alt = 'Снимок камеры';
    const canvas = el('canvas', 'sp-frame-overlay');
    canvas.id = 'cal-overlay';
    frameWrap.append(img, canvas);
    sideEl.appendChild(frameWrap);
    const btnRow = el('div', 'sp-row');
    const snapBtn = el('button', 'btn ghost', 'Снимок');
    snapBtn.onclick = onCameraChange;
    btnRow.appendChild(snapBtn);
    const gridBtn = el('button', 'btn ghost', 'Сетка 1 м');
    gridBtn.onclick = () => { gridOn = !gridOn; drawCalOverlay(); };
    btnRow.appendChild(gridBtn);
    sideEl.appendChild(btnRow);

    const list = el('div', 'sp-points');
    list.id = 'cal-points';
    sideEl.appendChild(list);

    const compute = el('button', 'btn primary', 'Калибровать');
    compute.id = 'cal-compute';
    compute.onclick = onCompute;
    sideEl.appendChild(compute);
    const result = el('div', 'sp-cal-result small');
    result.id = 'cal-result';
    sideEl.appendChild(result);

    sideEl.appendChild(buildMarkerSection());

    onCameraChange();
  }

  // ------------------------------------------------ авто-калибровка по маркерам

  function buildMarkerSection() {
    const box = el('div', 'sp-markers');
    // предзаполнение ID под стандартный лист (6 маркеров)
    if (!markerRows.length) {
      markerRows = [0, 1, 2, 3, 4, 5].map((id) => (
        { id, world: null, detected: false, sizePx: null }));
    }
    box.appendChild(el('h3', null, 'Авто-калибровка по маркерам'));
    const sheet = el('a', 'btn ghost', 'Скачать лист маркеров');
    sheet.href = '/api/spatial/markers/sheet?count=6&marker_cm=10';
    sheet.target = '_blank';
    sheet.title = 'Печать в масштабе 100%, контрольный отрезок 10 см';
    box.appendChild(sheet);
    const sheetPdf = el('a', 'btn ghost sp-link-block',
                        'PDF: 20 маркеров, по одному на страницу');
    sheetPdf.href = '/api/spatial/markers/sheet?count=20&marker_cm=18'
                  + '&one_per_page=true';
    sheetPdf.target = '_blank';
    sheetPdf.title = 'Крупные маркеры 18 см, A4, печать 100%';
    box.appendChild(sheetPdf);
    box.appendChild(el('div', 'muted small',
      'Распечатайте (масштаб 100%!), наклейте маркеры на пол с разбросом по ' +
      'кадру, замерьте центры маркеров от нуля координат — и заполните ' +
      'таблицу. Либо перенесите координаты с уже откалиброванной камеры.'));
    const detectBtn = el('button', 'btn ghost', 'Найти маркеры в кадре');
    detectBtn.onclick = onDetectMarkers;
    box.appendChild(detectBtn);
    const table = el('div', 'sp-points');
    table.id = 'marker-table';
    box.appendChild(table);
    // перенос координат с откалиброванной камеры (маркеры общие)
    const chainWrap = el('div', 'sp-row');
    const chainSel = el('select');
    chainSel.id = 'chain-camera';
    chainWrap.appendChild(chainSel);
    const chainBtn = el('button', 'btn ghost', 'Перенести координаты');
    chainBtn.title = 'Камера-источник должна видеть те же маркеры';
    chainBtn.onclick = onChainMeasure;
    chainWrap.appendChild(chainBtn);
    box.appendChild(chainWrap);
    const autoBtn = el('button', 'btn primary', 'Калибровать автоматически');
    autoBtn.id = 'auto-cal-btn';
    autoBtn.onclick = onAutoCalibrate;
    box.appendChild(autoBtn);
    const autoResult = el('div', 'sp-cal-result small');
    autoResult.id = 'auto-cal-result';
    box.appendChild(autoResult);
    renderMarkerTable();
    return box;
  }

  function currentCameraId() {
    const select = $('cal-camera');
    return select ? parseInt(select.value, 10) : null;
  }

  async function onDetectMarkers() {
    const cameraId = currentCameraId();
    if (!cameraId) return;
    const res = await App.api(`/api/spatial/cameras/${cameraId}/markers`);
    if (!res) return;
    detectedMarkers = (res.markers || []).map((m) => ({
      id: m.marker_id, pixel: m.pixel, size: m.size_px,
    }));
    for (const m of res.markers || []) {
      const row = markerRows.find((r) => r.id === m.marker_id);
      if (row) {
        row.detected = true; row.sizePx = m.size_px;
      } else {
        markerRows.push({ id: m.marker_id, world: null,
                          detected: true, sizePx: m.size_px });
      }
    }
    for (const r of markerRows) {
      if (!detectedMarkers.some((d) => d.id === r.id)) r.detected = false;
    }
    renderMarkerTable();
    drawCalOverlay();
    if (!res.markers || !res.markers.length) {
      App.toast('Маркеры не найдены: проверьте видимость, наклейку и блики');
    }
  }

  async function onChainMeasure() {
    const cameraId = currentCameraId();
    const srcSel = $('chain-camera');
    if (!cameraId || !srcSel || !srcSel.value) {
      App.toast('Выберите откалиброванную камеру-источник');
      return;
    }
    const res = await App.api(
      `/api/spatial/cameras/${srcSel.value}/markers/measure`, 'POST');
    if (!res) return;
    let moved = 0;
    for (const m of res.markers || []) {
      const row = markerRows.find((r) => r.id === m.marker_id);
      if (row) {
        row.world = m.world; moved++;
      } else {
        markerRows.push({ id: m.marker_id, world: m.world,
                          detected: false, sizePx: null });
      }
    }
    renderMarkerTable();
    App.toast(moved
      ? `Перенесены координаты ${moved} маркеров с камеры ${srcSel.value}`
      : 'Камера-источник сейчас не видит известных маркеров');
  }

  async function onAutoCalibrate() {
    const cameraId = currentCameraId();
    if (!cameraId) return;
    const pts = markerRows.filter((r) => r.world)
      .map((r) => ({ marker_id: r.id, world: r.world }));
    if (pts.length < 4) {
      App.toast('Нужны мировые координаты минимум 4 маркеров');
      return;
    }
    const res = await App.api(
      `/api/spatial/cameras/${cameraId}/calibration/auto`, 'POST',
      { points: pts });
    if (!res) return;
    calibration = res;
    gridOn = true;
    const result = $('auto-cal-result');
    if (result) {
      result.textContent =
        `Откалибрована автоматически. Ошибка: ${res.reprojection_error} м; ` +
        `маркеры: ${res.used_markers.join(', ')}` +
        (res.missing_markers.length
          ? `; не найдены: ${res.missing_markers.join(', ')}` : '');
    }
    drawCalOverlay();
    await App.loadWorld(App.floorId);
  }

  function renderMarkerTable() {
    const table = $('marker-table');
    if (!table) return;
    table.innerHTML = '';
    markerRows.sort((a, b) => a.id - b.id);
    for (const row of markerRows) {
      const line = el('div', 'sp-point-row');
      const idLabel = el('span', 'sp-point-idx', `ID ${row.id}`);
      if (row.detected) idLabel.classList.add('sp-marker-detected');
      line.appendChild(idLabel);
      line.appendChild(el('span', 'sp-point-px muted small',
        row.detected ? `виден (${row.sizePx?.toFixed(0)} px)` : 'не виден'));
      const wx = el('input', 'sp-point-w');
      wx.type = 'number'; wx.step = 'any'; wx.placeholder = 'X, м';
      wx.value = row.world ? row.world[0] : '';
      wx.oninput = () => setMarkerWorld(row, 0, wx.value);
      const wy = el('input', 'sp-point-w');
      wy.type = 'number'; wy.step = 'any'; wy.placeholder = 'Y, м';
      wy.value = row.world ? row.world[1] : '';
      wy.oninput = () => setMarkerWorld(row, 1, wy.value);
      line.append(wx, wy);
      if (row.world) line.appendChild(el('span', 'sp-point-ok', '✓'));
      table.appendChild(line);
    }
    // выбор камер для переноса координат
    const chainSel = $('chain-camera');
    if (chainSel) {
      const current = currentCameraId();
      chainSel.innerHTML = '';
      const calibrated = (App.world ? App.world.cameras : [])
        .filter((c) => c.calibrated && c.camera_id !== current);
      if (!calibrated.length) {
        const opt = el('option', null, 'нет откалиброванных камер');
        opt.value = '';
        chainSel.appendChild(opt);
      }
      for (const c of calibrated) {
        const opt = el('option', null, `с CAM ${c.camera_id}`);
        opt.value = c.camera_id;
        chainSel.appendChild(opt);
      }
    }
  }

  function setMarkerWorld(row, axis, value) {
    const v = parseFloat(value);
    if (isNaN(v)) {
      if (row.world) row.world = null;
    } else {
      if (!row.world) row.world = [0, 0];
      row.world[axis] = v;
    }
    renderMarkerTable();
  }

  async function onCameraChange() {
    const select = $('cal-camera');
    if (!select) return;
    points = []; calibration = null; gridOn = false;
    detectedMarkers = [];
    // world-координаты маркеров НЕ сбрасываем — они общие для всех камер,
    // меняется только факт видимости в кадре выбранной камеры
    for (const r of markerRows) { r.detected = false; r.sizePx = null; }
    renderMarkerTable();
    const cameraId = select.value;
    const img = $('cal-img');
    img.src = `/api/cameras/${cameraId}/snapshot?ts=${Date.now()}`;
    img.onerror = () => {
      $('cal-result').textContent =
        'Кадр недоступен: камера не подключена (нужен живой поток).';
    };
    img.onload = () => setupFrame();
    const res = await App.api(`/api/spatial/cameras/${cameraId}/calibration`);
    calibration = res && res.calibrated ? res : null;
    renderPoints();
    drawCalOverlay();
  }

  function setupFrame() {
    const img = $('cal-img'), canvas = $('cal-overlay');
    if (!img || !canvas) return;
    canvas.width = img.clientWidth || img.naturalWidth;
    canvas.height = img.clientHeight || img.naturalHeight;
    snapshot = { img, canvas, ctx: canvas.getContext('2d') };
    canvas.onclick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const px = (e.clientX - rect.left) * scaleX;
      const py = (e.clientY - rect.top) * scaleY;
      points.push({ pixel: [Math.round(px), Math.round(py)], world: null });
      renderPoints();
      drawCalOverlay();
    };
    drawCalOverlay();
  }

  function renderPoints() {
    const list = $('cal-points');
    if (!list) return;
    list.innerHTML = '';
    points.forEach((p, i) => {
      const row = el('div', 'sp-point-row');
      row.appendChild(el('span', 'sp-point-idx', `#${i + 1}`));
      row.appendChild(el('span', 'sp-point-px muted small',
        `(${p.pixel[0]}, ${p.pixel[1]})`));
      const wx = el('input', 'sp-point-w');
      wx.type = 'number'; wx.step = 'any'; wx.placeholder = 'X, м';
      wx.value = p.world ? p.world[0] : '';
      wx.oninput = () => setWorld(i, 0, wx.value);
      const wy = el('input', 'sp-point-w');
      wy.type = 'number'; wy.step = 'any'; wy.placeholder = 'Y, м';
      wy.value = p.world ? p.world[1] : '';
      wy.oninput = () => setWorld(i, 1, wy.value);
      row.append(wx, wy);
      const mapBtn = el('button', 'btn ghost sp-point-map', 'карта');
      mapBtn.title = 'Взять мировые координаты кликом по карте';
      mapBtn.onclick = () => {
        pickIndex = i;
        App.toast('Кликните точку на карте (нужен правильно привязанный этаж)');
      };
      row.appendChild(mapBtn);
      if (p.world) {
        row.appendChild(el('span', 'sp-point-ok', '✓'));
      }
      list.appendChild(row);
    });
    const ready = points.filter((p) => p.world).length;
    const compute = $('cal-compute');
    if (compute) {
      compute.disabled = ready < 4;
      compute.textContent = ready < 4
        ? `Калибровать (нужно ≥4 точек, есть ${ready})` : 'Калибровать';
    }
  }

  function setWorld(i, axis, value) {
    if (!points[i]) return;
    const v = parseFloat(value);
    if (isNaN(v)) {
      if (points[i].world) points[i].world = null;
      return;
    }
    if (!points[i].world) points[i].world = [0, 0];
    points[i].world[axis] = v;
    renderPoints();
  }

  async function onCompute() {
    const select = $('cal-camera'), img = $('cal-img');
    if (!select || !img) return;
    const payload = {
      points: points.filter((p) => p.world)
        .map((p) => ({ pixel: p.pixel, world: p.world })),
      resolution: [img.naturalWidth, img.naturalHeight],
    };
    if (payload.points.length < 4) return;
    const res = await App.api(
      `/api/spatial/cameras/${select.value}/calibration`, 'POST', payload);
    if (!res) return;
    calibration = res;
    gridOn = true;
    $('cal-result').textContent =
      `Гомография сохранена. Ошибка репроекции: ${res.reprojection_error} м` +
      (res.reprojection_error > 0.5 ? ' — проверьте точки!' : '');
    drawCalOverlay();
    await App.loadWorld(App.floorId);
  }

  function drawCalOverlay() {
    if (!snapshot || !snapshot.ctx) return;
    const { ctx, canvas, img } = snapshot;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;
    // найденные ArUco-маркеры (зелёные квадраты + ID)
    for (const m of detectedMarkers) {
      const x = m.pixel[0] * scaleX, y = m.pixel[1] * scaleY;
      const s = Math.max(10, (m.size || 40) * scaleX * 0.6);
      ctx.strokeStyle = '#2ecc71';
      ctx.lineWidth = 2;
      ctx.strokeRect(x - s / 2, y - s / 2, s, s);
      ctx.fillStyle = '#2ecc71';
      ctx.font = 'bold 11px Roboto, "Segoe UI", sans-serif';
      ctx.fillText(`ID ${m.id}`, x + s / 2 + 4, y - 4);
    }
    // точки
    points.forEach((p, i) => {
      const x = p.pixel[0] * scaleX, y = p.pixel[1] * scaleY;
      ctx.strokeStyle = p.world ? '#2ecc71' : '#f1c40f';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
      ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6);
      ctx.stroke();
      ctx.fillStyle = 'rgba(232,236,243,0.9)';
      ctx.font = '10px Roboto, "Segoe UI", sans-serif';
      ctx.fillText(`${i + 1}`, x + 8, y - 6);
    });
    // мировая сетка 1 м через H⁻¹ — визуальная проверка калибровки
    if (gridOn && calibration && calibration.homography) {
      const Hinv = invert3x3(calibration.homography);
      if (!Hinv) return;
      ctx.strokeStyle = 'rgba(79,140,255,0.35)';
      ctx.lineWidth = 1;
      const bounds = gridBounds();
      for (let gx = Math.floor(bounds.minx); gx <= bounds.maxx; gx++) {
        ctx.beginPath();
        let started = false;
        for (let t = 0; t <= 20; t++) {
          const wy = bounds.miny + (bounds.maxy - bounds.miny) * t / 20;
          const px = applyH(Hinv, gx, wy);
          if (!px) { started = false; continue; }
          const sx = px[0] * scaleX, sy = px[1] * scaleY;
          started ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
          started = true;
        }
        ctx.stroke();
      }
      for (let gy = Math.floor(bounds.miny); gy <= bounds.maxy; gy++) {
        ctx.beginPath();
        let started = false;
        for (let t = 0; t <= 20; t++) {
          const wx = bounds.minx + (bounds.maxx - bounds.minx) * t / 20;
          const px = applyH(Hinv, wx, gy);
          if (!px) { started = false; continue; }
          const sx = px[0] * scaleX, sy = px[1] * scaleY;
          started ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
          started = true;
        }
        ctx.stroke();
      }
    }
  }

  function gridBounds() {
    // диапазон мировых координат по точкам калибровки + запас
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const p of points) {
      if (!p.world) continue;
      minx = Math.min(minx, p.world[0]); maxx = Math.max(maxx, p.world[0]);
      miny = Math.min(miny, p.world[1]); maxy = Math.max(maxy, p.world[1]);
    }
    if (minx === Infinity) return { minx: 0, miny: 0, maxx: 10, maxy: 10 };
    return { minx: minx - 2, miny: miny - 2, maxx: maxx + 2, maxy: maxy + 2 };
  }

  // -------------------------------------------------------------- отладка

  function renderDebug() {
    sideEl.appendChild(el('h3', null, 'Calibration debug'));
    const wrap = el('label', 'sp-field');
    wrap.appendChild(el('span', 'sp-field-label', 'Камера'));
    wrap.appendChild(cameraSelect('dbg-camera', startDebugStream));
    sideEl.appendChild(wrap);
    sideEl.appendChild(el('div', 'muted small',
      'bbox → foot point (крестик) → мировые координаты. Те же точки — ' +
      'на карте. Нет координат — камера не откалибрована.'));

    const frameWrap = el('div', 'sp-frame-wrap');
    const img = el('img', 'sp-frame');
    img.id = 'dbg-img';
    img.alt = 'Живой поток';
    const canvas = el('canvas', 'sp-frame-overlay');
    canvas.id = 'dbg-overlay';
    frameWrap.append(img, canvas);
    sideEl.appendChild(frameWrap);

    const legend = el('div', 'sp-debug-list small muted');
    legend.id = 'dbg-list';
    sideEl.appendChild(legend);
    startDebugStream();
  }

  async function startDebugStream() {
    const select = $('dbg-camera');
    if (!select) return;
    const cameraId = parseInt(select.value, 10);
    if (live && live.timer) clearInterval(live.timer);
    const img = $('dbg-img'), canvas = $('dbg-overlay');
    if (!img) return;
    img.src = `/api/cameras/${cameraId}/stream`;
    img.onload = () => {
      canvas.width = img.clientWidth || 960;
      canvas.height = img.clientHeight || 540;
    };
    const calib = await App.api(`/api/spatial/cameras/${cameraId}/calibration`);
    live = {
      img, canvas, ctx: canvas.getContext('2d'),
      cameraId, homography: calib && calib.calibrated ? calib.homography : null,
      timer: null,
    };
    live.timer = setInterval(() => pollDebugDetections(), 500);
    pollDebugDetections();
  }

  async function pollDebugDetections() {
    if (!live) return;
    const data = await App.api(`/api/cameras/${live.cameraId}/detections`);
    if (!data) return;
    const { ctx, canvas, img, homography } = live;
    const mapPoints = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = img.naturalWidth ? canvas.width / img.naturalWidth : 1;
    const scaleY = img.naturalHeight ? canvas.height / img.naturalHeight : 1;
    for (const track of data.tracks || []) {
      const [x1, y1, x2, y2] = track.bbox;   // нормализованные 0..1
      const px1 = x1 * img.naturalWidth * scaleX;
      const py1 = y1 * img.naturalHeight * scaleY;
      const px2 = x2 * img.naturalWidth * scaleX;
      const py2 = y2 * img.naturalHeight * scaleY;
      ctx.strokeStyle = '#4f8cff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
      const label = `T${track.track_id}` + (track.global_id ? ` · G#${track.global_id}` : '');
      ctx.fillStyle = 'rgba(79,140,255,0.9)';
      ctx.font = 'bold 11px Roboto, "Segoe UI", sans-serif';
      ctx.fillText(label, px1, py1 - 4);
      // foot point
      const fx = (x1 + x2) / 2 * img.naturalWidth;
      const fy = y2 * img.naturalHeight;
      const sx = fx * scaleX, sy = fy * scaleY;
      ctx.strokeStyle = '#f1c40f';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx - 6, sy - 6); ctx.lineTo(sx + 6, sy + 6);
      ctx.moveTo(sx + 6, sy - 6); ctx.lineTo(sx - 6, sy + 6);
      ctx.stroke();
      // мировые координаты
      if (homography) {
        const w = applyH(homography, fx, fy);
        if (w) {
          const text = `(${w[0].toFixed(1)}, ${w[1].toFixed(1)})`;
          ctx.fillStyle = '#f1c40f';
          ctx.fillText(text, sx + 8, sy + 4);
          const cam = App.world.cameras.find((c) => c.camera_id === live.cameraId);
          mapPoints.push({
            x: w[0], y: w[1], floor_id: cam ? cam.floor_id : App.floorId,
            label: `G#${track.global_id ?? track.track_id}`,
          });
        }
      }
    }
    window.SpatialMap.setDebugPoints(mapPoints);
    const list = $('dbg-list');
    if (list) {
      list.innerHTML = '';
      if (!live.homography) {
        list.appendChild(el('div', 'sp-warn',
          'Камера не откалибрована — откалибруйте её в режиме «Калибровка».'));
      } else if (!mapPoints.length) {
        list.appendChild(el('div', null, 'Людей в кадре нет.'));
      }
      for (const p of mapPoints) {
        list.appendChild(el('div', null,
          `${p.label}: X=${p.x.toFixed(2)} Y=${p.y.toFixed(2)}`));
      }
    }
  }

  // --------------------------------------------------- клик по карте

  function onMapClick(w) {
    if (mode !== 'calibration' || pickIndex == null || !points[pickIndex]) {
      return false;
    }
    points[pickIndex].world = [+w.x.toFixed(2), +w.y.toFixed(2)];
    pickIndex = null;
    renderPoints();
    drawCalOverlay();
    return true;
  }

  // --------------------------------------------------------- линейка 3×3

  function invert3x3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    return [
      [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
      [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
      [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
    ];
  }

  function applyH(H, x, y) {
    const w = H[2][0] * x + H[2][1] * y + H[2][2];
    if (Math.abs(w) < 1e-9) return null;
    return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w,
            (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
  }

  window.SpatialCalibration = { activate, deactivate, onMapClick };
})();
