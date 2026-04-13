const state = { currentQuest: null };

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function refreshQuests() {
  const quests = await fetchJSON('/api/quests');
  const ul = document.getElementById('quest-list');
  ul.innerHTML = '';
  for (const q of quests) {
    const li = document.createElement('li');
    li.textContent = `${q.id} (${q.chapter_count} ch)`;
    li.classList.toggle('active', q.id === state.currentQuest);
    li.onclick = () => selectQuest(q.id);
    ul.appendChild(li);
  }
}

async function selectQuest(qid) {
  state.currentQuest = qid;
  document.getElementById('quest-title').textContent = qid;
  document.getElementById('action-input').disabled = false;
  document.querySelector('#action-form button').disabled = false;
  await Promise.all([refreshQuests(), refreshChapters(), refreshTraces()]);
}

async function refreshChapters() {
  if (!state.currentQuest) return;
  const chapters = await fetchJSON(`/api/quests/${state.currentQuest}/chapters`);
  const box = document.getElementById('chapters');
  box.innerHTML = '';
  for (let i = 0; i < chapters.length; i++) {
    const c = chapters[i];
    const isLast = i === chapters.length - 1;
    const el = document.createElement('div');
    el.className = 'chapter';
    el.innerHTML = `
      <div class="action">[${c.update_number}] ${escapeHtml(c.player_action || '')}</div>
      <div class="prose">${escapeHtml(c.prose)}</div>
    `;
    if (isLast && c.choices && c.choices.length > 0) {
      const bar = document.createElement('div');
      bar.className = 'choices-bar';
      c.choices.forEach((text, idx) => {
        const btn = document.createElement('button');
        btn.className = 'choice';
        btn.dataset.idx = idx + 1;
        btn.textContent = `${idx + 1}. ${text}`;
        btn.onclick = () => {
          const input = document.getElementById('action-input');
          input.value = text;
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
  box.scrollTop = box.scrollHeight;
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

document.getElementById('action-form').onsubmit = async (e) => {
  e.preventDefault();
  const input = document.getElementById('action-input');
  const action = input.value.trim();
  if (!action || !state.currentQuest) return;
  input.value = '';
  const status = document.getElementById('status');
  status.textContent = 'Generating chapter...';
  try {
    const r = await fetchJSON(`/api/quests/${state.currentQuest}/advance`, {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({action}),
    });
    status.textContent = `Done. outcome=${r.outcome}`;
    await Promise.all([refreshChapters(), refreshTraces(), refreshQuests()]);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
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

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

refreshQuests();
