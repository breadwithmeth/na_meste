/* Карта 2.5D-модели помещения: top-down и изометрия на одном canvas.

Экспортирует window.SpatialMap — движок отрисовки и ввода:
  setWorld / setFloor / setPeople / setTrail / setPrediction /
  setDebugPoints / setSelected / setIso / setOverlay / fit / render /
  worldFromEvent / hitTest
Никаких фреймворков — vanilla JS по конвенциям проекта.
*/
(() => {
  const ISO_COS = Math.cos(Math.PI / 6);
  const ISO_SIN = Math.sin(Math.PI / 6);
  const ISO_Z = 0.9;              // масштаб высоты Z в изометрии

  const COLORS = {
    wall: '#5b6b85', door: '#4f8cff', stairs: '#b8925b', elevator: '#9b7ede',
    zone: 'rgba(79,140,255,0.10)', zoneStroke: 'rgba(79,140,255,0.45)',
    restricted: 'rgba(231,76,60,0.12)', restrictedStroke: 'rgba(231,76,60,0.55)',
    entrance: 'rgba(46,204,113,0.14)', entranceStroke: 'rgba(46,204,113,0.6)',
    coverage: 'rgba(79,140,255,0.10)', coverageStroke: 'rgba(79,140,255,0.35)',
    node: '#8b96a8', edge: 'rgba(139,150,168,0.45)',
    trail: 'rgba(46,204,113,0.8)', predict: 'rgba(241,196,15,0.9)',
    debug: '#f1c40f',
  };
  const PERSON_COLORS = ['#4f8cff', '#2ecc71', '#f1c40f', '#e67e22',
    '#e74c3c', '#9b59b6', '#1abc9c', '#fd79a8', '#55efc4', '#74b9ff'];

  const NODE_ICONS = {
    corridor: '▬', room: '⌂', door: '⌸', stairs: '⩗',
    elevator: '⇕', entrance: '→', exit: '←', restricted_zone: '⛔',
  };

  const state = {
    canvas: null, ctx: null, dpr: 1,
    world: null, floorId: null,
    people: [], trail: null, prediction: null, debugPoints: [],
    selected: null, iso: false,
    view: { scale: 40, ox: 0, oy: 0 },
    floorplans: {},              // floorId -> {img, promise}
    overlay: null,               // drawFn(ctx) — незавершённые фигуры редактора
    hover: null,
    clickAt: null, dragStart: null, dragging: false,
  };

  // ------------------------------------------------------------- данные

  function floor() {
    if (!state.world) return null;
    return state.world.floors.find((f) => f.id === state.floorId) || null;
  }

  function floorZ(id) {
    const f = state.world && state.world.floors.find((x) => x.id === id);
    return f ? f.z : 0;
  }

  function personColor(gid) {
    return PERSON_COLORS[((gid || 0) % PERSON_COLORS.length + PERSON_COLORS.length) % PERSON_COLORS.length];
  }

  // ------------------------------------------------------- преобразования

  function toScreen(x, y, z) {
    const v = state.view;
    if (state.iso) {
      return {
        x: v.ox + (x - y) * ISO_COS * v.scale,
        y: v.oy + (x + y) * ISO_SIN * v.scale - (z || 0) * ISO_Z * v.scale,
      };
    }
    return { x: v.ox + x * v.scale, y: v.oy + y * v.scale };
  }

  function toWorld(sx, sy) {           // только для top-down
    const v = state.view;
    return { x: (sx - v.ox) / v.scale, y: (sy - v.oy) / v.scale };
  }

  function eventPos(e) {
    const rect = state.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function worldFromEvent(e) {
    return toWorld(eventPos(e).x, eventPos(e).y);
  }

  // -------------------------------------------------------------- fit

  function contentBounds() {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    const add = (x, y) => {
      minx = Math.min(minx, x); miny = Math.min(miny, y);
      maxx = Math.max(maxx, x); maxy = Math.max(maxy, y);
    };
    const floors = state.world ? state.world.floors : [];
    const inScope = state.iso ? floors : floors.filter((f) => f.id === state.floorId);
    for (const f of inScope) {
      for (const feat of f.features || []) {
        const g = feat.geometry || {};
        if (g.start && g.end) { add(g.start[0], g.start[1]); add(g.end[0], g.end[1]); }
        for (const p of g.points || []) add(p[0], p[1]);
      }
      if (f.floorplan_scale && f.floorplan_origin && state.floorplans[f.id]) {
        const img = state.floorplans[f.id].img;
        if (img) {
          const [ox, oy] = f.floorplan_origin;
          add(ox, oy); add(ox + img.naturalWidth * f.floorplan_scale,
                           oy + img.naturalHeight * f.floorplan_scale);
        }
      }
    }
    for (const cam of (state.world ? state.world.cameras : [])) {
      if (state.iso || cam.floor_id === state.floorId) {
        if (cam.position) add(cam.position.x, cam.position.y);
        for (const p of cam.coverage_polygon || []) add(p[0], p[1]);
      }
    }
    for (const n of (state.world ? state.world.nodes : [])) {
      if (state.iso || n.floor_id === state.floorId) add(n.position[0], n.position[1]);
    }
    for (const p of state.people) {
      if (p.x != null && (state.iso || p.floor_id === state.floorId)) add(p.x, p.y);
    }
    if (minx === Infinity) { add(0, 0); add(20, 20); }
    return { minx, miny, maxx, maxy };
  }

  function fit() {
    if (!state.canvas) return;
    const b = contentBounds();
    const w = state.canvas.clientWidth, h = state.canvas.clientHeight;
    const bw = Math.max(1, b.maxx - b.minx), bh = Math.max(1, b.maxy - b.miny);
    let scale = Math.min(w / (bw + 4), h / (bh + 4));
    scale = Math.max(2, Math.min(200, scale));
    state.view.scale = scale;
    // центрируем (в изометрии центр тоже считается по toScreen)
    const cx = (b.minx + b.maxx) / 2, cy = (b.miny + b.maxy) / 2;
    const center = toScreen(cx, cy, 0);
    state.view.ox += w / 2 - center.x;
    state.view.oy += h / 2 - center.y;
    render();
  }

  // --------------------------------------------------------- отрисовка

  function render() {
    if (!state.ctx) return;
    const ctx = state.ctx;
    const w = state.canvas.width, h = state.canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.scale(state.dpr, state.dpr);
    if (!state.world || !state.world.floors.length) {
      document.getElementById('sp-empty').hidden = false;
      ctx.restore();
      updateBadge();
      return;
    }
    document.getElementById('sp-empty').hidden = true;

    const floors = state.iso
      ? [...state.world.floors].sort((a, b) => a.z - b.z)
      : [floor()].filter(Boolean);
    for (const f of floors) drawFloor(ctx, f, f.id === state.floorId);

    if (state.overlay) {
      try { state.overlay(ctx, toScreen); } catch (err) { console.error(err); }
    }
    ctx.restore();
    updateBadge();
  }

  function updateBadge() {
    const badge = document.getElementById('sp-map-badge');
    if (!badge) return;
    const f = floor();
    const mode = state.iso ? 'изометрия' : 'top-down';
    badge.textContent = f
      ? `${f.name} · ${mode} · ${(state.view.scale).toFixed(0)} px/м`
      : mode;
  }

  function drawFloor(ctx, f, isCurrent) {
    ctx.globalAlpha = state.iso && !isCurrent ? 0.45 : 1.0;
    drawFloorplan(ctx, f);
    for (const feat of f.features || []) drawFeature(ctx, feat, f);
    for (const cam of state.world.cameras) {
      if (cam.floor_id !== f.id || !cam.position) continue;
      drawCamera(ctx, cam, f);
    }
    drawNav(ctx, f);
    drawPeople(ctx, f);
    drawDebugPoints(ctx, f);
    ctx.globalAlpha = 1.0;
  }

  function drawFloorplan(ctx, f) {
    if (!f.floorplan_scale || !f.floorplan_origin) return;
    const entry = state.floorplans[f.id];
    if (!entry || !entry.img || !entry.img.complete) {
      loadFloorplan(f);
      return;
    }
    const img = entry.img;
    const [ox, oy] = f.floorplan_origin;
    const s = f.floorplan_scale * state.view.scale;
    if (state.iso) {
      // четыре угла подложки в изометрии
      const corners = [[0, 0], [img.naturalWidth, 0],
                       [img.naturalWidth, img.naturalHeight], [0, img.naturalHeight]];
      const pts = corners.map(([ix, iy]) =>
        toScreen(ox + ix * f.floorplan_scale, oy + iy * f.floorplan_scale, f.z));
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.clip();
      // рисуем через transform: аффинная оценка (для планировки достаточно)
      const approx = affineForQuad(pts, img);
      if (approx) {
        ctx.transform(approx.a, approx.b, approx.c, approx.d, approx.e, approx.f);
        ctx.drawImage(img, 0, 0);
      }
      ctx.restore();
      return;
    }
    const p = toScreen(ox, oy);
    ctx.globalAlpha *= 0.85;
    ctx.drawImage(img, p.x, p.y, img.naturalWidth * s, img.naturalHeight * s);
    ctx.globalAlpha = state.iso ? 0.45 : 1.0;
  }

  function affineForQuad(pts, img) {
    // приближение: параллелограмм из левого верхнего угла + векторы сторон
    const [p0, p1, p3] = [pts[0], pts[1], pts[3]];
    const w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return null;
    return {
      a: (p1.x - p0.x) / w, b: (p1.y - p0.y) / w,
      c: (p3.x - p0.x) / h, d: (p3.y - p0.y) / h,
      e: p0.x, f: p0.y,
    };
  }

  function drawFeature(ctx, feat, f) {
    const g = feat.geometry || {};
    const z = f.z;
    const selected = state.selected && state.selected.kind === 'feature'
      && state.selected.id === feat.id;
    ctx.lineWidth = selected ? 3.5 : 2.5;
    if (g.start && g.end) {
      const color = COLORS[feat.type] || COLORS.wall;
      const a = toScreen(g.start[0], g.start[1], z);
      const b = toScreen(g.end[0], g.end[1], z);
      if (state.iso && feat.type === 'wall') {
        // стена — параллелограмм высотой height
        const height = g.height != null ? g.height : 3.0;
        const a2 = toScreen(g.start[0], g.start[1], z + height);
        const b2 = toScreen(g.end[0], g.end[1], z + height);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.lineTo(b2.x, b2.y); ctx.lineTo(a2.x, a2.y);
        ctx.closePath();
        ctx.fillStyle = 'rgba(91,107,133,0.30)';
        ctx.fill();
      }
      ctx.strokeStyle = selected ? '#ffffff' : color;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      ctx.stroke();
      if (feat.type === 'door') {
        // дверь — засечка в середине
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        ctx.beginPath();
        ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
      if (feat.type === 'stairs' || feat.type === 'elevator') {
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        ctx.fillStyle = color;
        ctx.font = '12px "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(feat.type === 'stairs' ? '⩗' : '⇕', mx, my - 6);
      }
      return;
    }
    if (g.points && g.points.length >= 3) {
      const isRestricted = feat.type === 'restricted_zone';
      const isEntrance = feat.type === 'entrance' || feat.type === 'exit';
      ctx.beginPath();
      g.points.forEach((p, i) => {
        const s = toScreen(p[0], p[1], z);
        i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y);
      });
      ctx.closePath();
      ctx.fillStyle = isRestricted ? COLORS.restricted
        : isEntrance ? COLORS.entrance : COLORS.zone;
      ctx.fill();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = selected ? '#ffffff'
        : isRestricted ? COLORS.restrictedStroke
        : isEntrance ? COLORS.entranceStroke : COLORS.zoneStroke;
      ctx.lineWidth = selected ? 2.5 : 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
      if (feat.name) {
        const c = g.points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0, 0]);
        const s = toScreen(c[0] / g.points.length, c[1] / g.points.length, z);
        ctx.fillStyle = 'rgba(232,236,243,0.75)';
        ctx.font = '11px "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(feat.name, s.x, s.y);
      }
    }
  }

  function drawCamera(ctx, cam, f) {
    const selected = state.selected && state.selected.kind === 'camera'
      && state.selected.id === cam.camera_id;
    const pos = cam.position;
    // зона видимости
    if (cam.coverage_polygon && cam.coverage_polygon.length >= 3) {
      ctx.beginPath();
      cam.coverage_polygon.forEach((p, i) => {
        const s = toScreen(p[0], p[1], f.z);
        i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y);
      });
      ctx.closePath();
      ctx.fillStyle = COLORS.coverage;
      ctx.fill();
      ctx.strokeStyle = COLORS.coverageStroke;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    const c = toScreen(pos.x, pos.y, state.iso ? f.z + pos.z : 0);
    // FOV-клин по yaw (в мире yaw 0 = +X, по часовой)
    if (cam.rotation) {
      const yaw = (cam.rotation.yaw * Math.PI) / 180;
      const half = ((cam.fov ? cam.fov.horizontal : 90) * Math.PI) / 360;
      const len = 4 * state.view.scale;
      ctx.beginPath();
      ctx.moveTo(c.x, c.y);
      const a1 = { x: c.x + Math.cos(yaw - half) * len, y: c.y + Math.sin(yaw - half) * len };
      const a2 = { x: c.x + Math.cos(yaw + half) * len, y: c.y + Math.sin(yaw + half) * len };
      ctx.lineTo(a1.x, a1.y);
      ctx.lineTo(a2.x, a2.y);
      ctx.closePath();
      ctx.fillStyle = 'rgba(79,140,255,0.08)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(79,140,255,0.30)';
      ctx.stroke();
      // направление взгляда
      ctx.beginPath();
      ctx.moveTo(c.x, c.y);
      ctx.lineTo(c.x + Math.cos(yaw) * len, c.y + Math.sin(yaw) * len);
      ctx.stroke();
    }
    // значок
    ctx.beginPath();
    ctx.arc(c.x, c.y, selected ? 8 : 6, 0, Math.PI * 2);
    ctx.fillStyle = cam.calibrated ? '#2ecc71' : '#6b7687';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = selected ? '#ffffff' : '#171d27';
    ctx.stroke();
    ctx.fillStyle = 'rgba(232,236,243,0.85)';
    ctx.font = 'bold 10px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`CAM ${cam.camera_id}${cam.calibrated ? '' : ' ⚠'}`, c.x, c.y - 10);
  }

  function drawNav(ctx, f) {
    const nodes = state.world.nodes.filter((n) => n.floor_id === f.id);
    const byId = {};
    for (const n of nodes) byId[n.id] = n;
    ctx.strokeStyle = COLORS.edge;
    ctx.lineWidth = 1.5;
    for (const e of state.world.edges) {
      const a = byId[e.from_node], b = byId[e.to_node];
      if (!a || !b) continue;
      // межэтажное ребро рисуем на текущем этаже пунктиром
      if (a.floor_id !== b.floor_id) {
        if (a.floor_id !== f.id && b.floor_id !== f.id) continue;
        ctx.setLineDash([2, 5]);
      }
      const sa = toScreen(a.position[0], a.position[1], f.z);
      const sb = toScreen(b.position[0], b.position[1], f.z);
      ctx.beginPath();
      ctx.moveTo(sa.x, sa.y);
      ctx.lineTo(sb.x, sb.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const n of nodes) {
      const selected = state.selected && state.selected.kind === 'node'
        && state.selected.id === n.id;
      const s = toScreen(n.position[0], n.position[1], f.z);
      ctx.beginPath();
      ctx.arc(s.x, s.y, selected ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = n.type === 'stairs' || n.type === 'elevator'
        ? '#b8925b' : COLORS.node;
      ctx.fill();
      if (selected) {
        ctx.lineWidth = 2; ctx.strokeStyle = '#ffffff'; ctx.stroke();
      }
      ctx.fillStyle = 'rgba(232,236,243,0.7)';
      ctx.font = '10px "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      const label = n.name || n.type;
      ctx.fillText(`${NODE_ICONS[n.type] || '•'} ${label}`, s.x, s.y - 9);
    }
  }

  function drawPeople(ctx, f) {
    for (const p of state.people) {
      if (p.x == null || p.y == null) continue;
      if (!state.iso && p.floor_id !== f.id) continue;
      if (state.iso && p.floor_id !== f.id) continue;
      const selected = state.selected && state.selected.kind === 'person'
        && state.selected.id === p.global_id;
      const color = personColor(p.global_id);
      // трейл выбранного
      if (selected && state.trail && state.trail.length > 1) {
        ctx.beginPath();
        let started = false;
        for (const t of state.trail) {
          if (t.x == null) continue;
          const s = toScreen(t.x, t.y, f.z);
          started ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y);
          started = true;
        }
        ctx.strokeStyle = COLORS.trail;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      // предсказание выбранного
      if (selected && state.prediction && state.prediction.expected_positions) {
        const base = state.prediction.position;
        ctx.beginPath();
        let s0 = toScreen(base.x, base.y, f.z);
        ctx.moveTo(s0.x, s0.y);
        for (const pt of state.prediction.expected_positions) {
          const s = toScreen(pt.x, pt.y, f.z);
          ctx.lineTo(s.x, s.y);
        }
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = COLORS.predict;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const c = toScreen(p.x, p.y, f.z);
      // стрелка направления (длина ~ скорости)
      if (p.speed != null && p.speed > 0.25 && p.direction != null) {
        const len = Math.min(30, 10 + p.speed * 12);
        ctx.beginPath();
        ctx.moveTo(c.x, c.y);
        const hx = c.x + Math.cos(p.direction) * len;
        const hy = c.y + Math.sin(p.direction) * len;
        ctx.lineTo(hx, hy);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        // наконечник
        ctx.beginPath();
        ctx.moveTo(hx, hy);
        ctx.lineTo(hx - Math.cos(p.direction - 0.4) * 6, hy - Math.sin(p.direction - 0.4) * 6);
        ctx.lineTo(hx - Math.cos(p.direction + 0.4) * 6, hy - Math.sin(p.direction + 0.4) * 6);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(c.x, c.y, selected ? 9 : 7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      if (selected) {
        ctx.lineWidth = 2.5; ctx.strokeStyle = '#ffffff'; ctx.stroke();
      }
      ctx.fillStyle = 'rgba(232,236,243,0.95)';
      ctx.font = 'bold 11px "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      const speedTxt = p.speed != null ? ` · ${p.speed.toFixed(1)} м/с` : '';
      ctx.fillText(`G#${p.global_id}${speedTxt}`, c.x, c.y - 13);
    }
  }

  function drawDebugPoints(ctx, f) {
    for (const dp of state.debugPoints) {
      if (dp.floor_id != null && dp.floor_id !== f.id) continue;
      const s = toScreen(dp.x, dp.y, f.z);
      ctx.strokeStyle = COLORS.debug;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(s.x - 6, s.y - 6); ctx.lineTo(s.x + 6, s.y + 6);
      ctx.moveTo(s.x + 6, s.y - 6); ctx.lineTo(s.x - 6, s.y + 6);
      ctx.stroke();
      ctx.fillStyle = COLORS.debug;
      ctx.font = '10px "Segoe UI", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(dp.label || '', s.x + 8, s.y - 4);
    }
  }

  function loadFloorplan(f) {
    if (state.floorplans[f.id]) return;
    const entry = { img: null, promise: null };
    state.floorplans[f.id] = entry;
    entry.promise = new Promise((resolve) => {
      const img = new Image();
      img.onload = () => { entry.img = img; render(); resolve(); };
      img.onerror = () => resolve();
      img.src = `/api/spatial/floors/${f.id}/floorplan`;
    });
  }

  // ---------------------------------------------------------- hit-test

  function distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const l2 = dx * dx + dy * dy;
    if (!l2) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * dx + (py - y1) * dy) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  function hitTest(e) {
    if (state.iso) return isoHitTest(e);
    const w = worldFromEvent(e);
    const tol = 10 / state.view.scale;
    // люди
    let best = null, bestD = tol;
    for (const p of state.people) {
      if (p.x == null || p.floor_id !== state.floorId) continue;
      const d = Math.hypot(p.x - w.x, p.y - w.y);
      if (d < bestD) { bestD = d; best = { kind: 'person', obj: p }; }
    }
    if (best) return best;
    // узлы
    for (const n of state.world.nodes) {
      if (n.floor_id !== state.floorId) continue;
      if (Math.hypot(n.position[0] - w.x, n.position[1] - w.y) < tol) {
        return { kind: 'node', obj: n };
      }
    }
    // камеры
    for (const cam of state.world.cameras) {
      if (!cam.position || cam.floor_id !== state.floorId) continue;
      if (Math.hypot(cam.position.x - w.x, cam.position.y - w.y) < tol) {
        return { kind: 'camera', obj: cam };
      }
    }
    // стены/двери
    for (const feat of (floor() ? floor().features : [])) {
      const g = feat.geometry || {};
      if (g.start && g.end) {
        if (distToSegment(w.x, w.y, g.start[0], g.start[1], g.end[0], g.end[1]) < tol) {
          return { kind: 'feature', obj: feat };
        }
      }
    }
    return null;
  }

  function isoHitTest(e) {
    const pos = eventPos(e);
    let best = null, bestD = 14;
    for (const p of state.people) {
      if (p.x == null || p.floor_id == null) continue;
      const f = state.world.floors.find((x) => x.id === p.floor_id);
      if (!f) continue;
      const s = toScreen(p.x, p.y, f.z);
      const d = Math.hypot(s.x - pos.x, s.y - pos.y);
      if (d < bestD) { bestD = d; best = { kind: 'person', obj: p }; }
    }
    return best;
  }

  // ------------------------------------------------------------ ввод

  function setupInput(canvas) {
    canvas.addEventListener('mousedown', (e) => {
      state.dragStart = { x: e.clientX, y: e.clientY, ox: state.view.ox, oy: state.view.oy };
      state.dragging = false;
    });
    window.addEventListener('mousemove', (e) => {
      if (!state.dragStart) return;
      const dx = e.clientX - state.dragStart.x, dy = e.clientY - state.dragStart.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) state.dragging = true;
      if (state.dragging) {
        state.view.ox = state.dragStart.ox + dx;
        state.view.oy = state.dragStart.oy + dy;
        render();
      }
    });
    window.addEventListener('mouseup', (e) => {
      const wasDrag = state.dragging;
      state.dragStart = null;
      state.dragging = false;
      if (wasDrag || e.target !== canvas) return;
      // клик без перетаскивания
      if (callbacks.onMapClick) callbacks.onMapClick(worldFromEvent(e), e);
    });
    canvas.addEventListener('dblclick', (e) => {
      if (callbacks.onMapDblClick) callbacks.onMapDblClick(worldFromEvent(e), e);
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const pos = eventPos(e);
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const old = state.view.scale;
      state.view.scale = Math.max(2, Math.min(300, old * factor));
      if (!state.iso) {
        // зум к курсору (top-down)
        const w = toWorld(pos.x, pos.y);
        const after = { x: pos.x - w.x * state.view.scale, y: pos.y - w.y * state.view.scale };
        state.view.ox = after.x; state.view.oy = after.y;
      } else {
        state.view.ox = pos.x - (pos.x - state.view.ox) * (state.view.scale / old);
        state.view.oy = pos.y - (pos.y - state.view.oy) * (state.view.scale / old);
      }
      render();
    }, { passive: false });
  }

  const callbacks = { onMapClick: null, onMapDblClick: null };

  // ------------------------------------------------------------- resize

  function resize() {
    if (!state.canvas) return;
    state.dpr = window.devicePixelRatio || 1;
    const w = state.canvas.clientWidth, h = state.canvas.clientHeight;
    state.canvas.width = Math.max(1, Math.round(w * state.dpr));
    state.canvas.height = Math.max(1, Math.round(h * state.dpr));
    render();
  }

  // -------------------------------------------------------------- API

  window.SpatialMap = {
    init(canvas, cbs) {
      state.canvas = canvas;
      state.ctx = canvas.getContext('2d');
      Object.assign(callbacks, cbs || {});
      setupInput(canvas);
      window.addEventListener('resize', resize);
      resize();
    },
    setWorld(world) { state.world = world; state.floorplans = {}; render(); },
    setFloor(floorId) { state.floorId = floorId; render(); },
    setPeople(people) { state.people = people || []; render(); },
    setTrail(points) { state.trail = points; render(); },
    setPrediction(pred) { state.prediction = pred; render(); },
    setDebugPoints(points) { state.debugPoints = points || []; render(); },
    setSelected(sel) { state.selected = sel; render(); },
    setIso(iso) { state.iso = iso; render(); },
    setOverlay(fn) { state.overlay = fn; render(); },
    fit,
    render,
    worldFromEvent,
    hitTest,
    get iso() { return state.iso; },
  };
})();
