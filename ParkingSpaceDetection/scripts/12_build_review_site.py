#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode


def confidence_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_manifest(manifest_path):
    rows = []
    cam_all_dir = manifest_path.parent.parent / "review_crops_cam" / "all"
    tile_cam_dir = manifest_path.parent.parent / "review_crops_cam" / "tile_heatmaps"
    geocoder = make_geocoder(manifest_path.parent.parent)
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conf = confidence_float(row.get("confidence"))
            crop_name = Path(row["crop_file"]).name
            cam_name = f"{Path(crop_name).stem}_cam.png"
            cam_path = cam_all_dir / cam_name
            tile_cam_name = f"{Path(crop_name).stem}_tile_cam.png"
            tile_cam_path = tile_cam_dir / tile_cam_name
            geo = geocoder(row) if geocoder else {}
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "crop": f"../review_crops/all/{crop_name}",
                    "cam": f"../review_crops_cam/all/{cam_name}" if cam_path.exists() else None,
                    "tile_cam": f"../review_crops_cam/tile_heatmaps/{tile_cam_name}" if tile_cam_path.exists() else None,
                    "crop_name": crop_name,
                    "cam_name": cam_name if cam_path.exists() else None,
                    "tile_cam_name": tile_cam_name if tile_cam_path.exists() else None,
                    "band": row["confidence_band"],
                    "confidence": conf,
                    "source_image": Path(row["source_image"]).name,
                    "source_label": Path(row["source_label"]).name,
                    "line_number": int(row["line_number"]),
                    "class_id": int(row["class_id"]),
                    "crop_box": [
                        int(row["crop_xmin"]),
                        int(row["crop_ymin"]),
                        int(row["crop_xmax"]),
                        int(row["crop_ymax"]),
                    ],
                    "obb_norm": [
                        float(row["x1_norm"]),
                        float(row["y1_norm"]),
                        float(row["x2_norm"]),
                        float(row["y2_norm"]),
                        float(row["x3_norm"]),
                        float(row["y3_norm"]),
                        float(row["x4_norm"]),
                        float(row["y4_norm"]),
                    ],
                    **geo,
                }
            )
    return sorted(rows, key=lambda r: (-1 if r["confidence"] is None else -r["confidence"], r["rank"]))


def make_geocoder(run_dir):
    try:
        import rasterio
        from pyproj import Transformer
    except ModuleNotFoundError:
        print("Warning: rasterio/pyproj unavailable; Street View links will be omitted.")
        return None

    tif_index = {}
    for tif_dir in [run_dir / "second_round" / "images", run_dir / "images"]:
        if tif_dir.exists():
            for tif_path in sorted(tif_dir.glob("*.tif")):
                tif_index.setdefault(tif_path.stem, tif_path)
            for tif_path in sorted(tif_dir.glob("*.tiff")):
                tif_index.setdefault(tif_path.stem, tif_path)

    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    transform_cache = {}

    def geocode(row):
        source_stem = Path(row["source_image"]).stem
        tif_path = tif_index.get(source_stem)
        if tif_path is None:
            return {}

        if tif_path not in transform_cache:
            with rasterio.open(tif_path) as dataset:
                transform_cache[tif_path] = dataset.transform

        center_px = sum(float(row[f"x{i}_px"]) for i in range(1, 5)) / 4
        center_py = sum(float(row[f"y{i}_px"]) for i in range(1, 5)) / 4
        rd_x, rd_y = transform_cache[tif_path] * (center_px, center_py)
        lon, lat = transformer.transform(rd_x, rd_y)
        street_view_url = "https://www.google.com/maps/@" + "?" + urlencode(
            {
                "api": "1",
                "map_action": "pano",
                "viewpoint": f"{lat:.7f},{lon:.7f}",
            }
        )
        return {
            "rd_x": rd_x,
            "rd_y": rd_y,
            "lat": lat,
            "lon": lon,
            "street_view_url": street_view_url,
            "map_url": f"https://www.google.com/maps/@{lat:.7f},{lon:.7f},18z",
        }

    return geocode


def write_index(output_dir):
    (output_dir / "index.html").write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>YOLO OBB Review Crops</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <header class="topbar">
      <div>
        <p class="eyebrow">Parking bay detections</p>
        <h1>Review crops</h1>
      </div>
      <div class="stats" id="stats"></div>
    </header>

    <main>
      <section class="controls" aria-label="Filters">
        <label class="search">
          <span>Search tile</span>
          <input id="search" type="search" placeholder="32.png, 101, predict label...">
        </label>
        <div class="segments" id="bandFilters" aria-label="Confidence filter">
          <button class="active" data-band="all">All</button>
          <button data-band="high_confidence">High</button>
          <button data-band="medium_confidence">Medium</button>
          <button data-band="low_confidence">Low</button>
        </div>
        <div class="segments" id="cardImageMode" aria-label="Crop image mode">
          <button data-mode="crop">Crop</button>
          <button class="active" data-mode="attention">Attention</button>
        </div>
        <label class="range">
          <span>Minimum confidence <b id="minConfValue">0.00</b></span>
          <input id="minConf" type="range" min="0" max="1" step="0.01" value="0">
        </label>
      </section>

      <section class="reviewLayout">
        <div class="cropPanel">
          <section class="grid" id="grid" aria-live="polite"></section>
          <p class="empty" id="empty" hidden>No crops match this filter.</p>
        </div>
        <aside class="tileViewer" aria-label="Full tile attention map">
          <div class="tileFrameRow">
            <div class="tileNavGroup">
              <button class="tileNav" id="prevTile" type="button" aria-label="Previous tile">&lt; tile</button>
              <button class="tileNav" id="prevParking" type="button" aria-label="Previous parking">&lt; parking</button>
            </div>
            <div class="tileFrame" id="tileFrame">
              <img id="tileImage" alt="" hidden>
              <svg id="tileOverlay" viewBox="0 0 1 1" preserveAspectRatio="xMidYMid meet" hidden>
                <polygon id="tileObb" points=""></polygon>
              </svg>
            </div>
            <div class="tileNavGroup">
              <a class="streetViewLink" id="streetViewLink" href="#" target="_blank" rel="noopener" hidden>
                Google Maps
              </a>
              <button class="tileNav" id="nextTile" type="button" aria-label="Next tile">&gt; tile</button>
              <button class="tileNav" id="nextParking" type="button" aria-label="Next parking">&gt; parking</button>
            </div>
          </div>
          <div class="tileViewerHeader">
            <div>
              <p class="eyebrow">Full tile</p>
              <strong id="selectedTitle">No crop selected</strong>
            </div>
            <span id="selectedConfidence"></span>
          </div>
        </aside>
      </section>
    </main>

    <template id="cardTemplate">
      <article class="card" tabindex="0">
        <div class="imageWrap">
          <img loading="lazy" alt="">
        </div>
        <div class="meta">
          <div>
            <strong class="rank"></strong>
            <span class="source"></span>
          </div>
          <span class="confidence"></span>
        </div>
        <dl class="details">
          <div><dt>Label</dt><dd class="label"></dd></div>
          <div><dt>Line</dt><dd class="line"></dd></div>
          <div><dt>Crop</dt><dd class="cropbox"></dd></div>
        </dl>
      </article>
    </template>

    <script src="app.js"></script>
  </body>
</html>
""",
        encoding="utf-8",
    )


def write_styles(output_dir):
    (output_dir / "styles.css").write_text(
        """:root {
  color-scheme: light;
  --paper: #f4f1ea;
  --ink: #171717;
  --muted: #6d6a62;
  --line: #d8d1c3;
  --panel: #fffdf8;
  --accent: #1d6b5f;
  --accent-2: #d84f2a;
  --shadow: 0 18px 45px rgba(30, 28, 23, 0.12);
}

* { box-sizing: border-box; }

[hidden] { display: none !important; }

body {
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  background: var(--paper);
  color: var(--ink);
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 4;
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  padding: 22px clamp(18px, 4vw, 46px);
  background: rgba(244, 241, 234, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(12px);
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--accent);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0;
}

h1 {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(34px, 5vw, 64px);
  line-height: 0.9;
  font-weight: 700;
}

.stats {
  display: grid;
  grid-template-columns: repeat(3, auto);
  gap: 10px;
  align-items: end;
  color: var(--muted);
  font-size: 13px;
  text-align: right;
}

.stats b {
  display: block;
  color: var(--ink);
  font-size: 24px;
}

main {
  padding: 22px clamp(18px, 4vw, 46px) 46px;
}

.controls {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto auto minmax(210px, 320px);
  gap: 14px;
  align-items: end;
  margin-bottom: 22px;
}

.search span,
.range span {
  display: block;
  margin-bottom: 7px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}

input[type="search"] {
  width: 100%;
  height: 42px;
  padding: 0 13px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--panel);
  color: var(--ink);
  font: inherit;
}

input[type="range"] {
  width: 100%;
  accent-color: var(--accent);
}

.segments {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.segments button {
  height: 42px;
  padding: 0 13px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--panel);
  color: var(--ink);
  font: inherit;
  cursor: pointer;
}

.segments button.active {
  border-color: var(--ink);
  background: var(--ink);
  color: var(--paper);
}

.reviewLayout {
  display: grid;
  grid-template-columns: minmax(280px, 0.55fr) minmax(560px, 1.45fr);
  gap: 18px;
  align-items: start;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px;
}

.card {
  min-width: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
  cursor: pointer;
  outline: none;
}

.card:hover,
.card:focus-visible {
  border-color: #aea693;
  box-shadow: var(--shadow);
}

.card.selected {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(29, 107, 95, 0.22), var(--shadow);
}

.imageWrap {
  display: grid;
  place-items: center;
  aspect-ratio: 2 / 3;
  background: var(--paper);
}

.imageWrap img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.meta {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 9px 6px;
  border-top: 1px solid var(--line);
}

.rank {
  display: block;
  font-size: 13px;
}

.source {
  color: var(--muted);
  font-size: 10px;
}

.confidence {
  align-self: start;
  padding: 2px 6px;
  border-radius: 999px;
  background: rgba(29, 107, 95, 0.1);
  color: var(--accent);
  font-weight: 700;
  font-size: 10px;
}

.confidence.low {
  background: rgba(216, 79, 42, 0.11);
  color: var(--accent-2);
}

.details {
  display: grid;
  gap: 1px;
  margin: 0;
  padding: 0 9px 9px;
  color: var(--muted);
  font-size: 9px;
}

.details div {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: 6px;
}

dt { color: #908a7c; }
dd {
  margin: 0;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.empty {
  padding: 30px;
  border: 1px dashed var(--line);
  color: var(--muted);
  text-align: center;
}

.tileViewer {
  position: sticky;
  top: 122px;
  min-width: 0;
}

.tileViewerHeader {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: start;
  min-height: 54px;
  margin-top: 10px;
}

.tileViewerHeader strong {
  display: block;
  font-size: 18px;
}

#selectedConfidence {
  color: var(--accent);
  font-weight: 700;
}

.tileFrame {
  position: relative;
  display: grid;
  place-items: center;
  justify-self: center;
  width: min(100%, 60vh);
  height: 60vh;
  min-height: 360px;
  max-height: 720px;
  aspect-ratio: 1 / 1;
  background: #ebe7de;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
}

.tileFrameRow {
  display: grid;
  grid-template-columns: 86px minmax(0, auto) 86px;
  gap: 12px;
  align-items: center;
  justify-content: center;
}

.tileNavGroup {
  display: grid;
  gap: 8px;
}

.tileNav {
  width: 86px;
  padding: 8px 9px;
  border: 1px solid rgba(23, 23, 23, 0.22);
  border-radius: 4px;
  background: rgba(255, 253, 248, 0.9);
  color: var(--ink);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}

.tileNav:hover {
  background: var(--ink);
  color: var(--paper);
}

.tileNav:disabled {
  opacity: 0.35;
  cursor: default;
}

.tileNav:disabled:hover {
  background: rgba(255, 253, 248, 0.9);
  color: var(--ink);
}

.tileFrame img,
.tileFrame svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.tileFrame svg {
  pointer-events: none;
}

#tileObb {
  fill: rgba(216, 79, 42, 0.05);
  stroke: #ff1f00;
  stroke-width: 5px;
  vector-effect: non-scaling-stroke;
}

.streetViewLink {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  width: 86px;
  min-height: 34px;
  padding: 0 11px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--panel);
  color: var(--accent);
  font-weight: 700;
  text-decoration: none;
}

@media (max-width: 1120px) {
  .reviewLayout {
    display: block;
  }

  .cropPanel {
    display: none;
  }

  .tileViewer {
    position: static;
    margin-top: 0;
  }

  .tileFrame {
    width: min(100%, 66vh);
    height: 66vh;
  }
}

@media (max-width: 760px) {
  .topbar,
  .controls {
    display: block;
  }

  .stats {
    grid-template-columns: repeat(3, 1fr);
    margin-top: 16px;
    text-align: left;
  }

  .segments,
  .range {
    margin-top: 12px;
  }

  .tileFrameRow {
    grid-template-columns: 1fr 1fr;
  }

  .tileFrame {
    grid-column: 1 / -1;
    grid-row: 1;
  }

  .tileNav {
    grid-row: 2;
    width: 100%;
  }

  .streetViewLink {
    width: 100%;
  }
}
""",
        encoding="utf-8",
    )


def write_app(output_dir, rows):
    data = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    (output_dir / "app.js").write_text(
        f"""const CROPS = {data};

const state = {{
  band: "all",
  query: "",
  minConf: 0,
  cardMode: "attention",
  selectedRank: null,
}};

const grid = document.querySelector("#grid");
const empty = document.querySelector("#empty");
const stats = document.querySelector("#stats");
const template = document.querySelector("#cardTemplate");
const search = document.querySelector("#search");
const minConf = document.querySelector("#minConf");
const minConfValue = document.querySelector("#minConfValue");
const bandFilters = document.querySelector("#bandFilters");
const cardImageMode = document.querySelector("#cardImageMode");
const selectedTitle = document.querySelector("#selectedTitle");
const selectedConfidence = document.querySelector("#selectedConfidence");
const tileImage = document.querySelector("#tileImage");
const tileOverlay = document.querySelector("#tileOverlay");
const tileObb = document.querySelector("#tileObb");
const prevTile = document.querySelector("#prevTile");
const nextTile = document.querySelector("#nextTile");
const prevParking = document.querySelector("#prevParking");
const nextParking = document.querySelector("#nextParking");
const streetViewLink = document.querySelector("#streetViewLink");
let visibleItems = [];

function fmtConfidence(value) {{
  return value === null ? "n/a" : value.toFixed(3);
}}

function cropBoxText(box) {{
  return box.join(", ");
}}

function tileSortValue(tileName) {{
  const stem = tileName.replace(/\\.[^.]+$/, "");
  const numeric = Number(stem);
  return Number.isFinite(numeric) ? numeric : stem;
}}

function matches(crop) {{
  const haystack = `${{crop.source_image}} ${{crop.source_label}} ${{crop.crop_name}}`.toLowerCase();
  const conf = crop.confidence ?? -1;
  return (state.band === "all" || crop.band === state.band)
    && haystack.includes(state.query)
    && conf >= state.minConf;
}}

function renderStats(items) {{
  const high = CROPS.filter(c => c.band === "high_confidence").length;
  const medium = CROPS.filter(c => c.band === "medium_confidence").length;
  stats.innerHTML = `
    <span><b>${{items.length}}</b>shown</span>
    <span><b>${{high}}</b>high</span>
    <span><b>${{medium}}</b>medium</span>
  `;
}}

function obbPoints(points) {{
  const pairs = [];
  for (let i = 0; i < points.length; i += 2) {{
    pairs.push(`${{points[i]}},${{points[i + 1]}}`);
  }}
  return pairs.join(" ");
}}

function clearSelection() {{
  selectedTitle.textContent = "No crop selected";
  selectedConfidence.textContent = "";
  tileImage.hidden = true;
  tileOverlay.setAttribute("hidden", "");
  tileImage.removeAttribute("src");
  tileObb.setAttribute("points", "");
  streetViewLink.hidden = true;
  streetViewLink.removeAttribute("href");
  streetViewLink.removeAttribute("data-map-url");
  updateNavButtons();
}}

function selectCrop(crop) {{
  state.selectedRank = crop.rank;
  selectedTitle.textContent = `${{crop.source_image}} - #${{String(crop.rank).padStart(3, "0")}}`;
  selectedConfidence.textContent = fmtConfidence(crop.confidence);
  if (crop.street_view_url) {{
    streetViewLink.href = crop.street_view_url;
    streetViewLink.dataset.mapUrl = crop.map_url || "";
    streetViewLink.hidden = false;
  }} else {{
    streetViewLink.hidden = true;
    streetViewLink.removeAttribute("href");
    streetViewLink.removeAttribute("data-map-url");
  }}
  updateNavButtons();

  if (!crop.tile_cam) {{
    tileImage.hidden = true;
    tileOverlay.setAttribute("hidden", "");
    tileImage.removeAttribute("src");
    return;
  }}

  tileImage.src = crop.tile_cam;
  tileImage.alt = `Full tile attention map for detection rank ${{crop.rank}} from ${{crop.source_image}}`;
  tileImage.hidden = false;
  tileObb.setAttribute("points", obbPoints(crop.obb_norm));
  tileOverlay.removeAttribute("hidden");
}}

function selectedIndex() {{
  return visibleItems.findIndex(crop => crop.rank === state.selectedRank);
}}

function tileNames() {{
  return [...new Set(visibleItems.map(crop => crop.source_image))]
    .sort((a, b) => {{
      const av = tileSortValue(a);
      const bv = tileSortValue(b);
      if (typeof av === "number" && typeof bv === "number") return av - bv;
      return String(av).localeCompare(String(bv), undefined, {{ numeric: true }});
    }});
}}

function selectedCrop() {{
  return visibleItems.find(crop => crop.rank === state.selectedRank) || null;
}}

function selectedTileName() {{
  const crop = selectedCrop();
  return crop ? crop.source_image : null;
}}

function parkingItemsForSelectedTile() {{
  const tile = selectedTileName();
  return tile ? visibleItems.filter(crop => crop.source_image === tile) : [];
}}

function updateNavButtons() {{
  const tile = selectedTileName();
  const tiles = tileNames();
  const tileIndex = tile ? tiles.indexOf(tile) : -1;
  prevTile.disabled = tileIndex <= 0;
  nextTile.disabled = tileIndex < 0 || tileIndex >= tiles.length - 1;

  const parkings = parkingItemsForSelectedTile();
  const parkingIndex = parkings.findIndex(crop => crop.rank === state.selectedRank);
  prevParking.disabled = parkingIndex <= 0;
  nextParking.disabled = parkingIndex < 0 || parkingIndex >= parkings.length - 1;
}}

function firstParkingInTile(tile) {{
  return visibleItems.find(crop => crop.source_image === tile) || null;
}}

function stepTile(direction) {{
  const tiles = tileNames();
  if (!tiles.length) return;
  const tile = selectedTileName();
  let index = tile ? tiles.indexOf(tile) : -1;
  if (index < 0) {{
    index = direction > 0 ? -1 : tiles.length;
  }}
  const nextIndex = Math.max(0, Math.min(tiles.length - 1, index + direction));
  const crop = firstParkingInTile(tiles[nextIndex]);
  if (crop) selectCrop(crop);
  render();
}}

function stepParking(direction) {{
  const parkings = parkingItemsForSelectedTile();
  if (!parkings.length) return;
  let index = parkings.findIndex(crop => crop.rank === state.selectedRank);
  if (index < 0) {{
    index = direction > 0 ? -1 : parkings.length;
  }}
  const nextIndex = Math.max(0, Math.min(parkings.length - 1, index + direction));
  selectCrop(parkings[nextIndex]);
  render();
}}

function render() {{
  const items = CROPS.filter(matches);
  visibleItems = items;
  if (state.selectedRank !== null && !items.some(crop => crop.rank === state.selectedRank)) {{
    state.selectedRank = null;
    clearSelection();
  }}
  if (state.selectedRank === null && items.length > 0) {{
    selectCrop(items[0]);
  }}
  grid.replaceChildren();
  empty.hidden = items.length !== 0;
  renderStats(items);

  for (const crop of items) {{
    const node = template.content.cloneNode(true);
    const card = node.querySelector(".card");
    const img = node.querySelector("img");
    const showAttention = state.cardMode === "attention" && crop.cam;
    img.src = showAttention ? crop.cam : crop.crop;
    img.alt = `${{showAttention ? "Cropped attention map" : "Normal crop"}} for detection rank ${{crop.rank}} from ${{crop.source_image}}`;
    card.dataset.rank = crop.rank;
    card.classList.toggle("selected", state.selectedRank === crop.rank);
    card.addEventListener("click", () => {{
      selectCrop(crop);
      render();
    }});
    card.addEventListener("keydown", event => {{
      if (event.key === "Enter" || event.key === " ") {{
        event.preventDefault();
        selectCrop(crop);
        render();
      }}
    }});
    node.querySelector(".rank").textContent = `#${{String(crop.rank).padStart(3, "0")}}`;
    node.querySelector(".source").textContent = crop.source_image;
    const conf = node.querySelector(".confidence");
    conf.textContent = fmtConfidence(crop.confidence);
    if ((crop.confidence ?? 0) < 0.5) conf.classList.add("low");
    node.querySelector(".label").textContent = crop.source_label;
    node.querySelector(".line").textContent = crop.line_number;
    node.querySelector(".cropbox").textContent = cropBoxText(crop.crop_box);
    grid.appendChild(node);
  }}
  updateNavButtons();
}}

search.addEventListener("input", event => {{
  state.query = event.target.value.trim().toLowerCase();
  render();
}});

minConf.addEventListener("input", event => {{
  state.minConf = Number(event.target.value);
  minConfValue.textContent = state.minConf.toFixed(2);
  render();
}});

bandFilters.addEventListener("click", event => {{
  const button = event.target.closest("button");
  if (!button) return;
  state.band = button.dataset.band;
  bandFilters.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === button));
  render();
}});

cardImageMode.addEventListener("click", event => {{
  const button = event.target.closest("button");
  if (!button) return;
  state.cardMode = button.dataset.mode;
  cardImageMode.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === button));
  render();
}});

prevTile.addEventListener("click", () => stepTile(-1));
nextTile.addEventListener("click", () => stepTile(1));
prevParking.addEventListener("click", () => stepParking(-1));
nextParking.addEventListener("click", () => stepParking(1));
streetViewLink.addEventListener("click", event => {{
  if (!event.shiftKey) return;
  const mapUrl = streetViewLink.dataset.mapUrl;
  if (!mapUrl) return;
  event.preventDefault();
  window.open(mapUrl, "_blank", "noopener");
}});

clearSelection();
render();
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build a static review site from YOLO crop manifest.csv.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    write_index(args.output)
    write_styles(args.output)
    write_app(args.output, rows)
    print(f"Wrote {len(rows)} crops to {args.output}")


if __name__ == "__main__":
    main()
