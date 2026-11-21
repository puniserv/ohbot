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
  level: (d) => ($("meter-bar").style.width = Math.min(100, d.level * 100) + "%"),
  settings: applySettings,
  doctor: (d) => showDoctor(d.results),
};

function connect() {
  ws = new WebSocket("ws://" + location.host + "/ws");
  ws.onopen = () => ($("offline").hidden = true);
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

document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    if (!e.repeat) talkStart();
  } else if (e.key === "Escape") {
    send({ cmd: "stop" });
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
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

connect();
