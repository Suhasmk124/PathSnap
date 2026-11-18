"use strict";
const get = id => document.getElementById(id);
const form = get("trip-form"), select = get("trip"), run = get("run"), canvas = get("map");
const ctx = canvas.getContext("2d");
const gpsColor = "#275dae", routeColor = "#0c806e";
let data, world, zoom = 1, dx = 0, dy = 0, dragging;
const timed = new Intl.NumberFormat("en", {maximumFractionDigits: 1});
const percentage = new Intl.NumberFormat("en", {style: "percent", maximumFractionDigits: 2});

function connection(state, text) {
  const el = get("connection"); el.dataset.state = state;
  el.replaceChildren(document.createElement("i"), text);
}
function error(text) {
  get("error").textContent = text || "";
  get("error").hidden = !text;
}
async function request(path, method = "GET") {
  const res = await fetch(path, {method, headers: {Accept: "application/json"}});
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed (" + res.status + ")");
  }
  return res.json();
}
async function init() {
  try {
    const trips = await request("/api/trips");
    select.replaceChildren(...trips.map(id => new Option(id, id)));
    if (!trips.length) throw new Error("No trips are available.");
    select.disabled = run.disabled = false; connection("ready", "API connected");
  } catch (e) {
    connection("error", "API unavailable"); error(e.message);
    select.replaceChildren(new Option("No trips loaded"));
  }
}
function bbox(points) {
  const extents = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [latitude, longitude] of points) {
    extents[0] = Math.min(extents[0], latitude); extents[1] = Math.min(extents[1], longitude);
    extents[2] = Math.max(extents[2], latitude); extents[3] = Math.max(extents[3], longitude);
  }
  return extents;
}
function useResult(result) {
  data = result;
  const points = [...data.gps, ...data.predicted_route.flatMap(segment => segment.coords)];
  if (points.length) {
    const [south, west, north, east] = bbox(points);
    const lat = (south + north) / 2, lon = (west + east) / 2, lonScale = Math.cos(lat * Math.PI / 180);
    world = {lat, lon, lonScale, width: (east - west) * lonScale, height: north - south};
    get("bounds").textContent = lat.toFixed(4) + "°, " + lon.toFixed(4) + "°";
  } else { world = null; get("bounds").textContent = ""; }
  get("placeholder").hidden = Boolean(world);
  get("map-subtitle").textContent = data.trip_id + " · " + data.n_points.toLocaleString() + " observations";
  const metrics = data.metrics;
  for (const key of ["top1_acc_hmm", "top1_acc_clf", "route_continuity", "candidate_miss_rate"]) {
    get(key).textContent = metrics[key] == null ? "—" : percentage.format(metrics[key]);
  }
  get("inference_ms").textContent = timed.format(metrics.inference_ms) + " ms";
  get("metric-status").textContent = metrics.top1_acc_hmm == null ? "Reference metrics unavailable" : "Calculated on the API server";
  fit();
}
form.addEventListener("submit", async event => {
  event.preventDefault(); error();
  const id = select.value;
  run.disabled = select.disabled = true; run.textContent = "Reconstructing…";
  get("metric-status").textContent = "Running inference…";
  try {
    useResult(await request("/api/predict/" + encodeURIComponent(id), "POST"));
    connection("ready", "API connected");
  } catch (e) {
    get("metric-status").textContent = "Prediction failed"; error(e.message);
  } finally {
    select.disabled = run.disabled = false; run.textContent = "Reconstruct route →";
  }
});
function fit() { zoom = 1; dx = dy = 0; draw(); }
function changeZoom(factor) { zoom = Math.max(0.7, Math.min(18, zoom * factor)); draw(); }
function draw() {
  const {width, height} = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#ecf3f2"; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9e7e4"; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = ((dx % 48) + 48) % 48; x < width; x += 48) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
  for (let y = ((dy % 48) + 48) % 48; y < height; y += 48) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
  ctx.stroke();
  if (!world) return;
  const scale = zoom * Math.min((width - 66) / Math.max(world.width, 1e-6),
                                (height - 66) / Math.max(world.height, 1e-6));
  function xy([lat, lon]) {
    return [width / 2 + (lon - world.lon) * world.lonScale * scale + dx,
            height / 2 - (lat - world.lat) * scale + dy];
  }
  if (get("gps-layer").checked) {
    ctx.fillStyle = gpsColor; ctx.globalAlpha = 0.53; ctx.beginPath();
    for (const point of data.gps) { const [x,y] = xy(point); ctx.moveTo(x + 2,y); ctx.arc(x,y,2,0,2*Math.PI); }
    ctx.fill(); ctx.globalAlpha = 1;
  }
  if (get("route-layer").checked) {
    ctx.strokeStyle = routeColor; ctx.lineWidth = 3.3; ctx.lineCap = "round"; ctx.beginPath();
    for (const {coords} of data.predicted_route) { ctx.moveTo(...xy(coords[0])); ctx.lineTo(...xy(coords[1])); }
    ctx.stroke();
  }
}
get("gps-layer").addEventListener("change", draw);
get("route-layer").addEventListener("change", draw);
get("reset").addEventListener("click", fit);
get("zoom-in").addEventListener("click", () => changeZoom(1.35));
get("zoom-out").addEventListener("click", () => changeZoom(1 / 1.35));
canvas.addEventListener("wheel", event => { if (world) {event.preventDefault(); changeZoom(event.deltaY < 0 ? 1.13 : 1/1.13);} }, {passive:false});
canvas.addEventListener("pointerdown", event => {
  if (world) { dragging = [event.clientX - dx, event.clientY - dy]; canvas.setPointerCapture(event.pointerId); }
});
canvas.addEventListener("pointermove", event => { if (dragging) { dx = event.clientX - dragging[0]; dy = event.clientY - dragging[1]; draw(); } });
for (const event of ["pointerup", "pointercancel"]) canvas.addEventListener(event, () => {dragging = null;});
new ResizeObserver(draw).observe(canvas); init();
