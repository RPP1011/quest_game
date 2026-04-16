const state = {
  currentQuest: null,
  config: null,
  chapters: [],
  generating: false,
  genStartedAt: null,
  genTimerId: null,
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
  document.getElementById('quest-title').textContent = humanizeTitle(qid);
  document.getElementById('header-context').textContent = humanizeTitle(qid);
  await Promise.all([refreshQuests(), refreshConfig(), refreshChapters(), refreshTraces(), refreshScene()]);
  renderHeroOrScene();
}

async function refreshConfig() {
  if (!state.currentQuest) { state.config = null; return; }
  try {
    state.config = await fetchJSON(`/api/quests/${state.currentQuest}/config`);
  } catch (_) {
    state.config = null;
  }
  if (state.config && state.config.genre) {
    document.getElementById('quest-genre').textContent = state.config.genre;
  } else {
    document.getElementById('quest-genre').textContent = '';
  }
}

function renderHeroOrScene() {
  const heroPanel = document.getElementById('hero-panel');
  const scenePanel = document.getElementById('scene-panel');
  const hasChapters = state.chapters && state.chapters.length > 0;
  if (hasChapters) {
    heroPanel.hidden = true;
    return;
  }
  // Empty quest: show hero from config
  const cfg = state.config || {};
  const premise = cfg.premise || '';
  document.getElementById('hero-premise').textContent = premise;
  // Themes
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
  // Cast
  fetchJSON(`/api/quests/${state.currentQuest}/scene`).then(s => {
    const charsField = document.getElementById('hero-chars-field');
    const locField = document.getElementById('hero-location-field');
    if (s.present_characters && s.present_characters.length) {
      document.getElementById('hero-chars').textContent = s.present_characters.join(', ');
      charsField.hidden = false;
    } else {
      charsField.hidden = true;
    }
    if (s.location) {
      document.getElementById('hero-location').textContent = s.location;
      locField.hidden = false;
    } else {
      locField.hidden = true;
    }
  }).catch(() => {});

  heroPanel.hidden = !premise && !themes.length;
  scenePanel.hidden = true;
}

async function refreshScene() {
  if (!state.currentQuest) return;
  // Only show scene panel when there are chapters; otherwise hero panel takes over.
  const hasChapters = state.chapters && state.chapters.length > 0;
  const panel = document.getElementById('scene-panel');
  if (!hasChapters) {
    panel.hidden = true;
    return;
  }
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
  } catch (_) {
    panel.hidden = true;
  }
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
    const heading = `Chapter ${c.update_number}`;
    el.innerHTML = `
      <div class="chapter-heading">${heading}</div>
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
  // Show traces panel only when there's at least one trace
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

function startGenerating() {
  state.generating = true;
  state.genStartedAt = Date.now();
  document.getElementById('generating-panel').hidden = false;
  document.getElementById('action-input').disabled = true;
  document.querySelector('#action-form button').disabled = true;
  document.getElementById('status').textContent = '';
  const tick = () => {
    const elapsed = (Date.now() - state.genStartedAt) / 1000;
    document.getElementById('gen-elapsed').textContent = formatTimer(elapsed);
  };
  tick();
  state.genTimerId = setInterval(tick, 1000);
}

function stopGenerating(message) {
  state.generating = false;
  if (state.genTimerId) { clearInterval(state.genTimerId); state.genTimerId = null; }
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
    stopGenerating(`Done in ${formatTimer(elapsed)} · outcome=${r.outcome}`);
    await Promise.all([refreshChapters(), refreshTraces(), refreshQuests(), refreshScene()]);
    renderHeroOrScene();
    // Scroll to the latest chapter
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

refreshQuests();
