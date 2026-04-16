const state = {
  currentQuest: null,
  config: null,
  chapters: [],
  generating: false,
  genStartedAt: null,
  genTimerId: null,
  liveTracePollId: null,
  liveTraceSeenTids: new Set(),
  activeLiveTid: null,
  worldTab: 'characters',
};

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function humanizeTitle(id) {
  return String(id || '').replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function formatTimer(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

async function refreshQuests() {
  const quests = await fetchJSON('/api/quests');
  const ul = document.getElementById('quest-list');
  ul.innerHTML = '';
  for (const q of quests) {
    const li = document.createElement('li');
    li.textContent = `${humanizeTitle(q.id)} (${q.chapter_count} ch)`;
    li.classList.toggle('active', q.id === state.currentQuest);
    li.onclick = () => selectQuest(q.id);
    ul.appendChild(li);
  }
}

async function selectQuest(qid) {
  state.currentQuest = qid;
  document.getElementById('no-quest-state').hidden = true;
  document.getElementById('quest-view').hidden = false;
  document.getElementById('world-toggle').hidden = false;
  document.getElementById('quest-title').textContent = humanizeTitle(qid);
  document.getElementById('header-context').textContent = humanizeTitle(qid);
  // Seed the set of known trace IDs so live polling ignores them (prior chapters)
  try {
    const traces = await fetchJSON(`/api/quests/${qid}/traces`);
    state.liveTraceSeenTids = new Set(traces.map(t => t.trace_id));
  } catch (_) { state.liveTraceSeenTids = new Set(); }
  await Promise.all([refreshQuests(), refreshConfig(), refreshChapters(), refreshTraces(), refreshScene()]);
  renderHeroOrScene();
  await refreshStartingActions();
}

async function refreshConfig() {
  if (!state.currentQuest) { state.config = null; return; }
  try {
    state.config = await fetchJSON(`/api/quests/${state.currentQuest}/config`);
  } catch (_) {
    state.config = null;
  }
  document.getElementById('quest-genre').textContent = (state.config && state.config.genre) || '';
}

async function refreshStartingActions() {
  const box = document.getElementById('starting-actions');
  const list = document.getElementById('starter-list');
  list.innerHTML = '';
  if (!state.currentQuest || state.chapters.length > 0) { box.hidden = true; return; }
  let suggestions = [];
  try {
    suggestions = await fetchJSON(`/api/quests/${state.currentQuest}/starting-actions`);
  } catch (_) { suggestions = []; }
  if (!suggestions.length) { box.hidden = true; return; }
  for (const s of suggestions) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'starter';
    btn.innerHTML = `
      <div class="starter-title">${escapeHtml(s.title || '')}</div>
      <div class="starter-desc">${escapeHtml(s.description || '')}</div>
    `;
    btn.onclick = () => {
      const input = document.getElementById('action-input');
      input.value = s.description || s.title || '';
      input.focus();
      input.scrollIntoView({behavior: 'smooth', block: 'center'});
    };
    list.appendChild(btn);
  }
  box.hidden = false;
}

function renderHeroOrScene() {
  const heroPanel = document.getElementById('hero-panel');
  const scenePanel = document.getElementById('scene-panel');
  const hasChapters = state.chapters && state.chapters.length > 0;
  if (hasChapters) {
    heroPanel.hidden = true;
    return;
  }
  const cfg = state.config || {};
  const premise = cfg.premise || '';
  document.getElementById('hero-premise').textContent = premise;
  const themes = (cfg.themes || []).map(t => typeof t === 'string' ? t : (t.proposition || t.id || ''));
  const themesField = document.getElementById('hero-themes-field');
  const themesUl = document.getElementById('hero-themes');
  themesUl.innerHTML = '';
  if (themes.length) {
    for (const t of themes) {
      const li = document.createElement('li');
      li.textContent = t;
      themesUl.appendChild(li);
    }
    themesField.hidden = false;
  } else {
    themesField.hidden = true;
  }
  fetchJSON(`/api/quests/${state.currentQuest}/scene`).then(s => {
    const charsField = document.getElementById('hero-chars-field');
    const locField = document.getElementById('hero-location-field');
    if (s.present_characters && s.present_characters.length) {
      document.getElementById('hero-chars').textContent = s.present_characters.join(', ');
      charsField.hidden = false;
    } else { charsField.hidden = true; }
    if (s.location) {
      document.getElementById('hero-location').textContent = s.location;
      locField.hidden = false;
    } else { locField.hidden = true; }
  }).catch(() => {});
  heroPanel.hidden = !premise && !themes.length;
  scenePanel.hidden = true;
}

async function refreshScene() {
  if (!state.currentQuest) return;
  const hasChapters = state.chapters && state.chapters.length > 0;
  const panel = document.getElementById('scene-panel');
  if (!hasChapters) { panel.hidden = true; return; }
  try {
    const s = await fetchJSON(`/api/quests/${state.currentQuest}/scene`);
    panel.hidden = false;
    document.getElementById('scene-location').textContent = s.location || 'Unknown';
    document.getElementById('scene-characters').textContent =
      s.present_characters.length ? s.present_characters.join(', ') : 'None';
    document.getElementById('scene-threads').textContent =
      s.plot_threads.length ? s.plot_threads.join('; ') : 'None';
    const recap = (s.recent_prose_tail || '').trim();
    document.getElementById('scene-recap').textContent = recap || '—';
  } catch (_) { panel.hidden = true; }
}

async function refreshChapters() {
  if (!state.currentQuest) return;
  state.chapters = await fetchJSON(`/api/quests/${state.currentQuest}/chapters`);
  const box = document.getElementById('chapters');
  box.innerHTML = '';
  for (let i = 0; i < state.chapters.length; i++) {
    const c = state.chapters[i];
    const isLast = i === state.chapters.length - 1;
    const el = document.createElement('article');
    el.className = 'chapter';
    el.innerHTML = `
      <div class="chapter-heading">Chapter ${c.update_number}</div>
      ${c.player_action ? `<div class="action">› ${escapeHtml(c.player_action)}</div>` : ''}
      <div class="prose">${escapeHtml(c.prose)}</div>
    `;
    if (isLast && c.choices && c.choices.length > 0) {
      const bar = document.createElement('div');
      bar.className = 'choices-bar';
      c.choices.forEach((choice, idx) => {
        const title = typeof choice === 'string' ? choice : (choice.title || '');
        const desc = typeof choice === 'object' ? (choice.description || '') : '';
        const tags = typeof choice === 'object' && Array.isArray(choice.tags) ? choice.tags : [];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'choice';
        btn.dataset.idx = idx + 1;
        const tagsHtml = tags.length
          ? `<div class="tags">${tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>`
          : '';
        btn.innerHTML = `<div class="title">${idx + 1}. ${escapeHtml(title)}</div>${desc ? `<div class="desc">${escapeHtml(desc)}</div>` : ''}${tagsHtml}`;
        btn.onclick = () => {
          const input = document.getElementById('action-input');
          input.value = title;
          document.getElementById('action-form').requestSubmit();
        };
        bar.appendChild(btn);
      });
      const writeIn = document.createElement('button');
      writeIn.type = 'button';
      writeIn.className = 'choice write-in';
      writeIn.textContent = 'Write-in...';
      writeIn.onclick = () => {
        const input = document.getElementById('action-input');
        input.value = '';
        input.focus();
      };
      bar.appendChild(writeIn);
      el.appendChild(bar);
    }
    box.appendChild(el);
  }
}

async function refreshTraces() {
  if (!state.currentQuest) return;
  const traces = await fetchJSON(`/api/quests/${state.currentQuest}/traces`);
  const ul = document.getElementById('trace-list');
  ul.innerHTML = '';
  for (const t of traces) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="outcome-${t.outcome}">●</span> ${t.stages.join('→')} <small>${t.total_latency_ms}ms</small>`;
    li.onclick = () => showTrace(t.trace_id);
    ul.appendChild(li);
  }
  document.querySelector('main').classList.toggle('traces-visible', traces.length > 0);
  document.getElementById('trace-panel').hidden = traces.length === 0;
}

async function showTrace(tid) {
  const t = await fetchJSON(`/api/quests/${state.currentQuest}/traces/${tid}`);
  const box = document.getElementById('trace-detail');
  box.innerHTML = `<h3>${t.trace_id.slice(0, 8)} · <span class="outcome-${t.outcome}">${t.outcome}</span></h3>`;
  for (const s of t.stages) {
    const d = document.createElement('details');
    d.className = 'stage';
    d.innerHTML = `
      <summary>${s.stage_name} · ${s.latency_ms}ms</summary>
      <pre>${escapeHtml(typeof s.parsed_output === 'string' ? s.parsed_output : JSON.stringify(s.parsed_output, null, 2))}</pre>
    `;
    box.appendChild(d);
  }
}

// ---------------------------------------------------------------------------
// Generation lifecycle + live trace polling
// ---------------------------------------------------------------------------

function renderLiveStages(trace) {
  const ul = document.getElementById('gen-stages-list');
  ul.innerHTML = '';
  if (!trace || !trace.stages || !trace.stages.length) {
    const li = document.createElement('li');
    li.className = 'stage-empty';
    li.textContent = 'waiting for first stage…';
    ul.appendChild(li);
    return;
  }
  for (const s of trace.stages) {
    const li = document.createElement('li');
    const hasError = s.errors && s.errors.length > 0;
    li.className = hasError ? 'stage-error' : 'stage-done';
    const latSecs = (s.latency_ms || 0) / 1000;
    li.innerHTML = `
      <span class="stage-name">${escapeHtml(s.stage_name)}</span>
      <span class="stage-lat">${latSecs.toFixed(1)}s${hasError ? ' · error' : ''}</span>
    `;
    ul.appendChild(li);
  }
}

async function pollLiveTrace() {
  if (!state.generating || !state.currentQuest) return;
  try {
    // If we haven't pinned a live trace id yet, look for the first new trace
    if (!state.activeLiveTid) {
      const traces = await fetchJSON(`/api/quests/${state.currentQuest}/traces`);
      const fresh = traces.find(t => !state.liveTraceSeenTids.has(t.trace_id));
      if (fresh) { state.activeLiveTid = fresh.trace_id; }
    }
    if (state.activeLiveTid) {
      const t = await fetchJSON(`/api/quests/${state.currentQuest}/traces/${state.activeLiveTid}`);
      renderLiveStages(t);
    }
  } catch (_) { /* ignore polling hiccups */ }
}

function startGenerating() {
  state.generating = true;
  state.genStartedAt = Date.now();
  state.activeLiveTid = null;
  document.getElementById('generating-panel').hidden = false;
  document.getElementById('action-input').disabled = true;
  document.querySelector('#action-form button').disabled = true;
  document.getElementById('status').textContent = '';
  renderLiveStages(null);
  const tick = () => {
    const elapsed = (Date.now() - state.genStartedAt) / 1000;
    document.getElementById('gen-elapsed').textContent = formatTimer(elapsed);
  };
  tick();
  state.genTimerId = setInterval(tick, 1000);
  state.liveTracePollId = setInterval(pollLiveTrace, 2000);
  pollLiveTrace();
}

function stopGenerating(message) {
  state.generating = false;
  if (state.genTimerId) { clearInterval(state.genTimerId); state.genTimerId = null; }
  if (state.liveTracePollId) { clearInterval(state.liveTracePollId); state.liveTracePollId = null; }
  document.getElementById('generating-panel').hidden = true;
  document.getElementById('action-input').disabled = false;
  document.querySelector('#action-form button').disabled = false;
  document.getElementById('status').textContent = message || '';
}

document.getElementById('action-form').onsubmit = async (e) => {
  e.preventDefault();
  if (state.generating) return;
  const input = document.getElementById('action-input');
  const action = input.value.trim();
  if (!action || !state.currentQuest) return;
  input.value = '';
  startGenerating();
  try {
    const r = await fetchJSON(`/api/quests/${state.currentQuest}/advance`, {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({action}),
    });
    const elapsed = (Date.now() - state.genStartedAt) / 1000;
    // Register the just-finished trace id as "seen" so it isn't picked up next time
    if (r.trace_id) state.liveTraceSeenTids.add(r.trace_id);
    stopGenerating(`Done in ${formatTimer(elapsed)} · outcome=${r.outcome}`);
    await Promise.all([refreshChapters(), refreshTraces(), refreshQuests(), refreshScene()]);
    renderHeroOrScene();
    await refreshStartingActions();
    const box = document.getElementById('chapters');
    const last = box.lastElementChild;
    if (last) last.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (err) {
    stopGenerating(`Error: ${err.message}`);
  }
};

document.getElementById('new-quest-btn').onclick = async () => {
  const id = prompt('Quest id (e.g. tavern-01):');
  if (!id) return;
  const seedText = prompt('Seed JSON:', '{"entities": [{"id": "alice", "entity_type": "character", "name": "Alice"}]}');
  if (!seedText) return;
  let seed;
  try { seed = JSON.parse(seedText); } catch { alert('invalid JSON'); return; }
  try {
    await fetchJSON('/api/quests', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({id, seed}),
    });
    await refreshQuests();
    selectQuest(id);
  } catch (err) { alert(err.message); }
};

// ---------------------------------------------------------------------------
// World drawer
// ---------------------------------------------------------------------------

let cachedWorld = null;

function openWorldDrawer() {
  document.getElementById('world-backdrop').hidden = false;
  document.getElementById('world-drawer').hidden = false;
  if (!cachedWorld) loadWorld();
  else renderWorldTab(state.worldTab);
}
function closeWorldDrawer() {
  document.getElementById('world-backdrop').hidden = true;
  document.getElementById('world-drawer').hidden = true;
}

async function loadWorld() {
  if (!state.currentQuest) return;
  try {
    cachedWorld = await fetchJSON(`/api/quests/${state.currentQuest}/world`);
  } catch (err) {
    document.getElementById('world-content').innerHTML = `<p class="muted">Failed to load world: ${escapeHtml(err.message)}</p>`;
    return;
  }
  renderWorldTab(state.worldTab);
}

function renderWorldTab(tab) {
  state.worldTab = tab;
  document.querySelectorAll('.drawer-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tab);
  });
  const box = document.getElementById('world-content');
  if (!cachedWorld) { box.innerHTML = '<p class="muted">Loading…</p>'; return; }

  const renderEntity = (e) => `
    <div class="world-item">
      <div class="world-item-header">
        <span class="world-item-name">${escapeHtml(e.name)}</span>
        <span class="world-item-id">${escapeHtml(e.id)}</span>
        <span class="world-item-status ${e.status}">${e.status}</span>
      </div>
      ${e.role ? `<div class="world-item-role">${escapeHtml(e.role)}</div>` : ''}
      ${e.description ? `<div class="world-item-desc">${escapeHtml(e.description)}</div>` : ''}
    </div>
  `;

  const renderSimple = (items, label) => {
    if (!items || !items.length) return `<p class="muted">No ${label} seeded.</p>`;
    return items.map(renderEntity).join('');
  };

  const entityTabMap = {
    characters: 'character', factions: 'faction', locations: 'location',
    items: 'item', concepts: 'concept',
  };
  if (tab in entityTabMap) {
    const type = entityTabMap[tab];
    const items = (cachedWorld.entities_by_type || {})[type] || [];
    box.innerHTML = items.length
      ? items.map(renderEntity).join('')
      : `<p class="muted">No ${tab} seeded.</p>`;
    return;
  }

  if (tab === 'threads') {
    const items = cachedWorld.plot_threads || [];
    box.innerHTML = items.length ? items.map(t => `
      <div class="world-item">
        <div class="world-item-header">
          <span class="world-item-name">${escapeHtml(t.name)}</span>
          <span class="world-item-id">${escapeHtml(t.id)}</span>
          <span class="world-item-status ${t.status}">${t.status}</span>
        </div>
        <div class="world-item-desc">${escapeHtml(t.description)}</div>
        <div class="world-item-meta">priority ${t.priority} · arc: ${escapeHtml(t.arc_position)}${t.involved_entities.length ? ' · involves ' + t.involved_entities.map(escapeHtml).join(', ') : ''}</div>
      </div>
    `).join('') : '<p class="muted">No plot threads seeded.</p>';
    return;
  }

  if (tab === 'hooks') {
    const items = cachedWorld.foreshadowing || [];
    box.innerHTML = items.length ? items.map(h => `
      <div class="world-item">
        <div class="world-item-header">
          <span class="world-item-name">${escapeHtml(h.id)}</span>
          <span class="world-item-status ${h.status}">${h.status}</span>
        </div>
        <div class="world-item-desc">${escapeHtml(h.description)}</div>
        <div class="world-item-meta">planted @ update ${h.planted_at_update} · payoff target: ${escapeHtml(h.payoff_target || '—')}</div>
      </div>
    `).join('') : '<p class="muted">No foreshadowing hooks seeded.</p>';
    return;
  }

  if (tab === 'rules') {
    const items = cachedWorld.rules || [];
    box.innerHTML = items.length ? items.map(r => `
      <div class="world-item">
        <div class="world-item-header">
          <span class="world-item-name">${escapeHtml(r.id)}</span>
          <span class="world-item-id">[${escapeHtml(r.category)}]</span>
        </div>
        <div class="world-item-desc">${escapeHtml(r.description)}</div>
      </div>
    `).join('') : '<p class="muted">No world rules seeded.</p>';
    return;
  }

  if (tab === 'motifs') {
    const items = cachedWorld.motifs || [];
    box.innerHTML = items.length ? items.map(m => `
      <div class="world-item">
        <div class="world-item-header">
          <span class="world-item-name">${escapeHtml(m.name)}</span>
          <span class="world-item-id">${escapeHtml(m.id)}</span>
        </div>
        <div class="world-item-desc">${escapeHtml(m.description)}</div>
        ${m.semantic_range && m.semantic_range.length ? `<div class="world-item-meta">Semantic range: ${m.semantic_range.map(escapeHtml).join(', ')}</div>` : ''}
      </div>
    `).join('') : '<p class="muted">No motifs seeded.</p>';
    return;
  }
}

document.getElementById('world-toggle').onclick = openWorldDrawer;
document.getElementById('world-close').onclick = closeWorldDrawer;
document.getElementById('world-backdrop').onclick = closeWorldDrawer;
document.querySelectorAll('.drawer-tab').forEach(t => {
  t.onclick = () => renderWorldTab(t.dataset.tab);
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeWorldDrawer();
});

refreshQuests();
