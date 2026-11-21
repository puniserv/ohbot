"use strict";
const $ = (id) => document.getElementById(id);
const MARK = { true: "✔", false: "✘", null: "–" };
const STATE_PL = { idle: "Czeka", listening: "Słucha", transcribing: "Rozpoznaje", thinking: "Myśli", speaking: "Mówi", error: "Błąd" };
const ERRORS = {
  quota: "Brak środków na koncie OpenAI – doładuj konto",
  empty: "Nie usłyszałem – przytrzymaj przycisk dłużej",
  moderated: "Porozmawiajmy o czymś innym",
  nokey: "Brak klucza API – dodaj OPENAI_API_KEY do pliku .env",
  auth: "Klucz API odrzucony – sprawdź plik .env",
  network: "Brak internetu – robot nie może się połączyć",
};
const MAX_BUBBLES = 20;
const GESTURE_PL = { yes: "Tak", no: "Nie", surprised: "Zdziwienie", happy: "Radość", thinking: "Namysł" };
const MOTORS_PL = ["Kiwanie głową", "Obrót głowy", "Oczy w bok", "Powieki", "Górna warga", "Dolna warga", "Oczy góra/dół"];
const MOVE_EVERY_MS = 100; // a slider fires dozens of events per second; the serial link needs far fewer

let ws = null;
let colours = {};
let held = false;
let handsFree = false;
let toastTimer = null;
let barge = { enabled: false, threshold: 0.15 };

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function css(c) {
  return "rgb(" + c.map((v) => Math.round(v * 25.5)).join(",") + ")";
}

function bubble(kind, who, text) {
  const el = document.createElement("div");
  el.className = "bubble " + kind;
  const small = document.createElement("small");
  small.textContent = who;
  el.append(small, document.createTextNode(text));
  const box = $("subtitles");
  box.append(el);
  while (box.children.length > MAX_BUBBLES) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

function setActive(names) {
  document.querySelectorAll("#pipeline li").forEach((li) => li.classList.toggle("active", names.includes(li.dataset.stage)));
}

function setState(state) {
  $("robot-state").textContent = STATE_PL[state] || state;
  if (colours[state]) $("robot-dot").style.background = css(colours[state]);
  else if (state === "transcribing" && colours.thinking) $("robot-dot").style.background = css(colours.thinking);
  setActive(state === "listening" ? ["mic"] : state === "speaking" ? ["tts", "robot"] : []);
}

function setRobot(connected) {
  const b = $("robot-badge");
  b.textContent = connected ? "Robot: połączony" : "Robot: brak";
  b.classList.toggle("ok", connected);
}

function setCamera(ok) {
  $("face-error").hidden = ok !== false;
}

function applySettings(s) {
  $("mode").value = s.mode;
  $("mode-cost").hidden = s.mode === "ptt";
  handsFree = s.mode !== "ptt"; // the mic listens all the time, so the button only interrupts the robot
  $("talk").textContent = handsFree ? "Przerwij" : "Przytrzymaj i mów";
  if (handsFree) talkStop();
  if ([...$("persona").options].some((o) => o.value === s.persona)) $("persona").value = s.persona;
  if (!s.barge_in) return; // the app's own fallback to ptt sends only the mode
  barge = s.barge_in;
  $("barge-on").checked = barge.enabled;
  $("barge-threshold").value = barge.threshold;
}

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 8000);
}

function showDoctor(results) {
  const ul = $("doctor-results");
  ul.replaceChildren();
  for (const r of results) {
    const li = document.createElement("li");
    li.textContent = MARK[r.ok] + " " + r.name + ": " + r.detail;
    if (r.ok !== true && r.hint) {
      const h = document.createElement("span");
      h.className = "hint";
      h.textContent = r.hint;
      li.append(h);
    }
    ul.append(li);
  }
  $("doctor-panel").hidden = false;
}

const handlers = {
  snapshot(d) {
    colours = d.colours;
    $("persona").replaceChildren(...d.personas.map((p) => new Option(p, p)));
    applySettings(d);
    setState(d.state);
    setRobot(d.robot);
    setCamera(d.camera);
    $("nokey").hidden = d.has_key;
    if (d.mic_error) toast(d.mic_error);
  },
  state: (d) => setState(d.state),
  stage: (d) => setActive(d.status === "start" ? [d.name] : []),
  heard: (d) => {
    $("info").textContent = "";
    bubble("child", "Ty:", d.text);
  },
  info: (d) => ($("info").textContent = d.text),
  reply: (d) => bubble("robot", "Ohbot:", d.text),
  error: (d) => toast(ERRORS[d.kind] || (d.kind === "other" && d.message) || "Coś poszło nie tak – szczegóły w logu"),
  robot: (d) => setRobot(d.connected),
  camera: (d) => setCamera(d.ok),
  face: (d) => {
    const dot = $("face-dot");
    dot.classList.toggle("seen", d.visible);
    if (d.visible) {
      dot.style.left = 50 + d.x * 50 + "%";
      dot.style.top = 50 - d.y * 50 + "%"; // y is up
    }
  },
  level: (d) => ($("meter-bar").style.width = Math.min(100, d.level * 100) + "%"),
  settings: applySettings,
  doctor: (d) => showDoctor(d.results),
};

function connect() {
  ws = new WebSocket("ws://" + location.host + "/ws");
  ws.onopen = () => {
    $("offline").hidden = true;
    if (!$("test-panel").hidden) send({ cmd: "test_panel", open: true }); // the server forgets it on disconnect
  };
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (handlers[e.type]) handlers[e.type](e.data);
  };
  ws.onclose = () => {
    $("offline").hidden = false;
    setTimeout(connect, 2000);
  };
}

function talkStart() {
  if (handsFree) return send({ cmd: "stop" });
  if (held) return;
  held = true;
  $("talk").classList.add("held");
  send({ cmd: "ptt_start" });
}

function talkStop() {
  if (!held) return;
  held = false;
  $("talk").classList.remove("held");
  send({ cmd: "ptt_stop" });
}

const talk = $("talk");
talk.addEventListener("pointerdown", (e) => {
  talk.setPointerCapture(e.pointerId);
  talkStart();
});
["pointerup", "pointercancel", "lostpointercapture"].forEach((ev) => talk.addEventListener(ev, talkStop));
talk.addEventListener("contextmenu", (e) => e.preventDefault());

const typing = (e) => e.target.matches("input[type=text], input[type=number], textarea");

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && typing(e)) return;
  if (e.code === "Space") {
    e.preventDefault();
    if (!e.repeat) talkStart();
  } else if (e.key === "Escape") {
    send({ cmd: "stop" });
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && !typing(e)) {
    e.preventDefault();
    talkStop();
  }
});

window.addEventListener("blur", talkStop);
document.addEventListener("change", (e) => e.target.blur()); // a focused control would swallow Space
document.addEventListener("click", (e) => e.target.tagName === "BUTTON" && e.target.blur());

$("stop").onclick = () => send({ cmd: "stop" });
$("new").onclick = () => {
  $("subtitles").replaceChildren();
  send({ cmd: "new" });
};
$("mode").onchange = (e) => send({ cmd: "mode", value: e.target.value });
$("persona").onchange = (e) => {
  $("subtitles").replaceChildren();
  send({ cmd: "persona", value: e.target.value });
};
const sendBarge = () => send({ cmd: "barge_in", enabled: $("barge-on").checked, threshold: Number($("barge-threshold").value) });
$("barge-on").onchange = sendBarge;
$("barge-threshold").onchange = sendBarge;
$("doctor").onclick = () => {
  showDoctor([{ ok: null, name: "Diagnostyka", detail: "trwa sprawdzanie…" }]);
  send({ cmd: "doctor" });
};
$("doctor-close").onclick = () => ($("doctor-panel").hidden = true);

async function api(method, path, body) {
  const r = await fetch(path, { method, headers: body ? { "content-type": "application/json" } : {}, body: body && JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}
const fail = (e) => toast("Nie udało się: " + e.message);

const fields = () => document.querySelectorAll("#settings-panel [data-key]");
const fieldValue = (el) => (el.type === "checkbox" ? el.checked : el.type === "range" || el.type === "number" ? Number(el.value) : el.value);
function showValue(el) {
  const out = el.nextElementSibling;
  if (out && out.tagName === "OUTPUT") out.textContent = el.value;
}

let promptPersona = "";
let promptSaved = "";

async function openSettings() {
  const d = await api("GET", "/api/settings");
  $("voice").replaceChildren(...d.voices.map((v) => new Option(v, v)));
  for (const el of fields()) {
    const [section, key] = el.dataset.key.split(".");
    const v = d.values[section][key];
    if (el.type === "checkbox") el.checked = v;
    else el.value = v;
    showValue(el);
  }
  promptPersona = $("persona").value;
  $("prompt-persona").textContent = "(" + promptPersona + ")";
  promptSaved = (await api("GET", "/api/persona/" + encodeURIComponent(promptPersona))).text;
  $("prompt").value = promptSaved;
  $("settings-panel").hidden = false;
}

for (const el of fields()) {
  el.addEventListener("input", () => showValue(el));
  el.addEventListener("change", () => {
    const [section, key] = el.dataset.key.split(".");
    api("POST", "/api/settings", { [section]: { [key]: fieldValue(el) } }).catch(fail);
  });
}

$("settings").onclick = () => openSettings().catch(fail);
$("settings-close").onclick = () => ($("settings-panel").hidden = true);
$("voice-sample").onclick = () => send({ cmd: "voice_sample", voice: $("voice").value });
$("settings-save").onclick = async () => {
  try {
    await api("POST", "/api/settings/save");
    const text = $("prompt").value;
    if (text !== promptSaved) {
      await api("PUT", "/api/persona/" + encodeURIComponent(promptPersona), { text });
      promptSaved = text;
    }
    toast("Zapisano");
  } catch (e) {
    fail(e);
  }
};

const pendingMoves = new Map();
let moveTimer = null;
function queueMove(motor, pos) {
  pendingMoves.set(motor, pos);
  moveTimer ??= setTimeout(() => {
    moveTimer = null;
    for (const [m, p] of pendingMoves) send({ cmd: "motor", motor: m, pos: p });
    pendingMoves.clear();
  }, MOVE_EVERY_MS);
}

MOTORS_PL.forEach((name, motor) => {
  const input = Object.assign(document.createElement("input"), { type: "range", min: 0, max: 10, step: 0.1, value: motor === 3 ? 10 : 5 });
  input.oninput = () => queueMove(motor, Number(input.value));
  const label = document.createElement("label");
  label.append(name, input);
  $("motors").append(label);
});

$("test").onclick = async () => {
  try {
    const d = await api("GET", "/api/settings");
    $("gestures").replaceChildren(
      ...d.gestures.map((g) => {
        const b = document.createElement("button");
        b.textContent = GESTURE_PL[g] || g;
        b.onclick = () => send({ cmd: "gesture", name: g });
        return b;
      }),
    );
    $("test-panel").hidden = false;
    send({ cmd: "test_panel", open: true });
  } catch (e) {
    fail(e);
  }
};
$("test-close").onclick = () => {
  $("test-panel").hidden = true;
  send({ cmd: "test_panel", open: false });
};
$("close-mouth").onclick = () => send({ cmd: "close_mouth" });
$("neutral").onclick = () => send({ cmd: "neutral" });
$("say").onclick = () => {
  const text = $("say-text").value.trim();
  if (text) send({ cmd: "say", text });
};

connect();
