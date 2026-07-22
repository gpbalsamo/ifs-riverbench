#!/usr/bin/env python3
"""
Build a compact, single-page HTML dashboard for CaMa-Flood vs GloFAS benchmark plots.

The dashboard scans an output directory produced by benchmark_cmf_vs_glofas.py and builds
an index with:
  * rows = domains: Global + continents
  * columns = metrics/groups: Mean error, RMSE, NSE, KGE, differences, relative differences
  * separate sections for time-series and map plots
  * inline thumbnails
  * click-to-zoom modal/lightbox without opening a new page per plot
  * filters for experiment, domain, plot type and metric

Expected benchmark layout, for example:
  OUTDIR/
    plots_timeseries/*.png
    plots_maps/*.png
    plots_maps/continents/*.png
    benchmark_daily_domains_*.csv
    benchmark_daily_vs_control_*.csv

Example:
  python3 build_benchmark_html_index.py \
      --outdir /perm/pad/benchmark_2016 \
      --title "CaMa-Flood vs GloFAS benchmark — 2016" \
      --open
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DOMAINS = [
    "Global",
    "Africa",
    "Europe",
    "Asia",
    "NorthAmerica",
    "SouthAmerica",
    "Oceania",
]

TIMESERIES_METRIC_GROUPS = [
  ("daily_spatial_diagnostics", "Daily diagnostics"),
  ("mae", "MAE"),
  ("control_metric_differences", "Differences vs control"),
  ("relative_rmse_improvement", "Relative RMSE improvement"),
]

MAP_METRIC_GROUPS = [
  ("mean_error", "Mean-error"),
  ("rmse", "RMSE"),
  ("correlation", "Correlation"),
  ("nse", "NSE"),
  ("kge", "KGE"),
  ("mean_error_difference", "Δ Mean-error"),
  ("rmse_difference", "Δ RMSE"),
  ("correlation_difference", "Δ Correlation"),
  ("nse_difference", "Δ NSE"),
  ("kge_difference", "Δ KGE"),
  ("relative_rmse_improvement", "Relative RMSE improvement"),
]

METRIC_GROUPS = TIMESERIES_METRIC_GROUPS + MAP_METRIC_GROUPS

# More permissive aliases so the dashboard survives small filename changes.
METRIC_ALIASES = {
  "daily_spatial_diagnostics": ["timeseries_metrics_"],
  "mae": ["timeseries_mae_", "_mae_"],
  "control_metric_differences": ["timeseries_metric_differences_vs_control_"],
    "mean_error": ["mean_error_map", "mean_error_"],
    "rmse": ["rmse_map", "rmse_"],
  "correlation": ["correlation_map", "correlation_"],
    "nse": ["nse_map", "nse_"],
    "kge": ["kge_map", "kge_"],
    "mean_error_difference": ["mean_error_difference"],
    "rmse_difference": ["rmse_difference"],
  "correlation_difference": ["correlation_difference"],
    "nse_difference": ["nse_difference"],
    "kge_difference": ["kge_difference"],
    "relative_rmse_improvement": [
        "relative_rmse_improvement",
        "rel_rmse_improvement",
        "timeseries_relative_rmse_improvement",
    ],
}

TIMESERIES_PATTERNS = [
    re.compile(r"^timeseries_metrics_(?P<domain>[A-Za-z]+)\.png$"),
  re.compile(r"^timeseries_mae_(?P<domain>[A-Za-z]+)\.png$"),
  re.compile(r"^timeseries_metric_differences_vs_control_(?P<domain>[A-Za-z]+)\.png$"),
    re.compile(r"^timeseries_relative_rmse_improvement_(?P<domain>[A-Za-z]+)\.png$"),
]

EXPVER_RE = re.compile(r"(?P<expver>[a-z][a-z0-9]{3})", re.IGNORECASE)


@dataclass(frozen=True)
class PlotItem:
    path: Path
    relpath: str
    kind: str          # timeseries, map-global, map-continent
    domain: str
    metric: str
    label: str
    expver: str
    filename: str


def rel_href(path: Path, html_path: Path) -> str:
    return Path(os.path.relpath(path, html_path.parent)).as_posix()


def infer_metric(filename: str) -> tuple[str, str]:
    lower = filename.lower()
    # Order matters: differences before base metrics.
    for key, label in METRIC_GROUPS:
        for alias in METRIC_ALIASES.get(key, [key]):
            if alias in lower:
                return key, label
    return "other", "Other"


def infer_expver(filename: str) -> str:
    # Prefer tokens after known patterns.
    patterns = [
        r"map_(?P<exp>[a-z][a-z0-9]{3})(?:_|\.)",
        r"difference_(?P<exp>[a-z][a-z0-9]{3})_minus_",
        r"improvement_map_(?P<exp>[a-z][a-z0-9]{3})_vs_",
        r"improvement_(?P<exp>[a-z][a-z0-9]{3})_vs_",
    ]
    for pat in patterns:
        m = re.search(pat, filename, flags=re.IGNORECASE)
        if m:
            return m.group("exp")
    m = EXPVER_RE.search(filename)
    return m.group("expver") if m else "all"


def infer_domain_from_filename(filename: str, fallback: str = "Global") -> str:
    stem = Path(filename).stem
    for d in DOMAINS:
        if d != "Global" and (stem.endswith("_" + d) or ("_" + d + "_") in stem):
            return d
    for pat in TIMESERIES_PATTERNS:
        m = pat.match(filename)
        if m:
            return m.group("domain")
    return fallback


def scan_plots(outdir: Path, html_path: Path) -> list[PlotItem]:
    items: list[PlotItem] = []

    # Time series plots.
    ts_dir = outdir / "plots_timeseries"
    if ts_dir.exists():
        for p in sorted(ts_dir.glob("*.png")):
            metric, label = infer_metric(p.name)
            domain = infer_domain_from_filename(p.name)
            items.append(
                PlotItem(
                    path=p,
                    relpath=rel_href(p, html_path),
                    kind="timeseries",
                    domain=domain,
                    metric=metric,
                    label=label if metric != "other" else "Metrics time series",
                    expver="all",
                    filename=p.name,
                )
            )

    # Global map plots.
    maps_dir = outdir / "plots_maps"
    if maps_dir.exists():
        for p in sorted(maps_dir.glob("*.png")):
            metric, label = infer_metric(p.name)
            items.append(
                PlotItem(
                    path=p,
                    relpath=rel_href(p, html_path),
                    kind="map-global",
                    domain="Global",
                    metric=metric,
                    label=label,
                    expver=infer_expver(p.name),
                    filename=p.name,
                )
            )

    # Continental map plots.
    cont_dir = outdir / "plots_maps" / "continents"
    if cont_dir.exists():
        for p in sorted(cont_dir.glob("*.png")):
            metric, label = infer_metric(p.name)
            items.append(
                PlotItem(
                    path=p,
                    relpath=rel_href(p, html_path),
                    kind="map-continent",
                    domain=infer_domain_from_filename(p.name, fallback="Unknown"),
                    metric=metric,
                    label=label,
                    expver=infer_expver(p.name),
                    filename=p.name,
                )
            )

    return items


def unique_ordered(values: Iterable[str], preferred: list[str] | None = None) -> list[str]:
    vals = list(dict.fromkeys(v for v in values if v))
    if preferred:
        out = [v for v in preferred if v in vals]
        out.extend(v for v in vals if v not in out)
        return out
    return vals


def metric_order_key(metric: str) -> int:
    keys = [k for k, _ in METRIC_GROUPS]
    try:
        return keys.index(metric)
    except ValueError:
        return len(keys) + 1


def build_plot_card(item: PlotItem) -> str:
    title = f"{item.domain} — {item.label}"
    if item.expver not in ("all", ""):
        title += f" — {item.expver}"
    alt = html.escape(title)
    data = {
        "src": item.relpath,
        "title": title,
        "filename": item.filename,
        "domain": item.domain,
        "metric": item.metric,
        "kind": item.kind,
        "expver": item.expver,
    }
    return f"""
      <button class=\"plot-card\" type=\"button\"
        data-kind=\"{html.escape(item.kind)}\"
        data-domain=\"{html.escape(item.domain)}\"
        data-metric=\"{html.escape(item.metric)}\"
        data-expver=\"{html.escape(item.expver)}\"
        data-plot='{html.escape(json.dumps(data), quote=True)}'
        onclick=\"openLightbox(this)\">
        <div class=\"plot-title\">{html.escape(title)}</div>
        <img loading=\"lazy\" src=\"{html.escape(item.relpath)}\" alt=\"{alt}\">
        <div class=\"plot-file\">{html.escape(item.filename)}</div>
      </button>
    """


def build_matrix(items: list[PlotItem], kind: str, domains: list[str], metrics: list[tuple[str, str]]) -> str:
    subset = [i for i in items if i.kind == kind]
    if not subset:
        return f"<p class='empty'>No {html.escape(kind)} plots found.</p>"

    # Each cell can contain multiple experiment cards.
    cell: dict[tuple[str, str], list[PlotItem]] = {}
    for item in subset:
        cell.setdefault((item.domain, item.metric), []).append(item)

    rows = []
    header = "<tr><th class='sticky-left'>Area domain</th>" + "".join(
        f"<th>{html.escape(label)}</th>" for key, label in metrics
    ) + "</tr>"

    for domain in domains:
        tds = [f"<th class='sticky-left domain-name'>{html.escape(domain)}</th>"]
        for key, label in metrics:
            plots = sorted(cell.get((domain, key), []), key=lambda x: (x.expver, x.filename))
            if plots:
                tds.append("<td>" + "".join(build_plot_card(p) for p in plots) + "</td>")
            else:
                tds.append("<td class='missing'>—</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")

    return f"<div class='matrix-wrap'><table class='matrix'>{header}{''.join(rows)}</table></div>"


def _present_metrics_for_kind(
  items: list[PlotItem],
  kind: str,
  preferred_groups: list[tuple[str, str]],
) -> list[tuple[str, str]]:
  present = {i.metric for i in items if i.kind == kind}
  metrics = [(k, label) for k, label in preferred_groups if k in present]
  if "other" in present:
    metrics.append(("other", "Other"))
  return metrics


def build_html(outdir: Path, title: str, items: list[PlotItem], html_path: Path) -> str:
    domains = unique_ordered([i.domain for i in items], preferred=DOMAINS)
    expvers = unique_ordered([i.expver for i in items if i.expver != "all"])
    ts_metrics = _present_metrics_for_kind(items, "timeseries", TIMESERIES_METRIC_GROUPS)
    map_global_metrics = _present_metrics_for_kind(items, "map-global", MAP_METRIC_GROUPS)
    map_continent_metrics = _present_metrics_for_kind(items, "map-continent", MAP_METRIC_GROUPS)

    total = len(items)
    ts = sum(i.kind == "timeseries" for i in items)
    mg = sum(i.kind == "map-global" for i in items)
    mc = sum(i.kind == "map-continent" for i in items)

    ts_matrix = build_matrix(items, "timeseries", domains, ts_metrics)
    global_matrix = build_matrix(items, "map-global", ["Global"], map_global_metrics)
    continent_matrix = build_matrix(items, "map-continent", [d for d in domains if d != "Global"], map_continent_metrics)

    exp_options = "".join(f"<option value='{html.escape(e)}'>{html.escape(e)}</option>" for e in expvers)
    domain_options = "".join(f"<option value='{html.escape(d)}'>{html.escape(d)}</option>" for d in domains)
    all_metric_options = []
    seen_metrics = set()
    for metric_list in (ts_metrics, map_global_metrics, map_continent_metrics):
      for k, label in metric_list:
        if k not in seen_metrics:
          seen_metrics.add(k)
          all_metric_options.append((k, label))
    metric_options = "".join(f"<option value='{html.escape(k)}'>{html.escape(label)}</option>" for k, label in all_metric_options)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #0f172a; --panel: #111827; --card: #ffffff; --ink: #0f172a;
  --muted: #64748b; --line: #dbe3ef; --accent: #2563eb; --soft: #f8fafc;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background: #f1f5f9; color: var(--ink); }}
header {{ background: linear-gradient(135deg, #0f172a, #1e3a8a); color: white; padding: 22px 28px; }}
h1 {{ margin: 0; font-size: 26px; letter-spacing: -0.02em; }}
.subtitle {{ margin-top: 6px; opacity: 0.85; font-size: 14px; }}
.stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }}
.stat {{ background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; padding: 8px 12px; }}
.stat b {{ display: block; font-size: 20px; }}
.controls {{ position: sticky; top: 0; z-index: 20; background: rgba(248,250,252,0.96); backdrop-filter: blur(8px); padding: 12px 24px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
.controls label {{ font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 3px; }}
select, input {{ border: 1px solid #cbd5e1; border-radius: 9px; padding: 7px 9px; background: white; min-width: 140px; }}
button.small {{ border: 0; border-radius: 9px; padding: 8px 12px; background: var(--accent); color: white; cursor: pointer; }}
main {{ padding: 20px 24px 60px; }}
section {{ margin: 18px 0 28px; }}
section h2 {{ margin: 0 0 10px; font-size: 20px; }}
.note {{ color: var(--muted); font-size: 13px; margin: 0 0 12px; }}
.matrix-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 14px; background: white; box-shadow: 0 6px 20px rgba(15,23,42,0.06); }}
table.matrix {{ border-collapse: separate; border-spacing: 0; width: 100%; min-width: 1100px; }}
.matrix th {{ position: sticky; top: 0; background: #eaf0f8; z-index: 4; border-bottom: 1px solid var(--line); padding: 10px; font-size: 12px; text-align: center; }}
.matrix td {{ vertical-align: top; border-bottom: 1px solid #eef2f7; border-right: 1px solid #eef2f7; padding: 8px; min-width: 220px; max-width: 280px; }}
.sticky-left {{ position: sticky !important; left: 0; z-index: 6 !important; background: #eaf0f8 !important; min-width: 130px; }}
.domain-name {{ text-align: left !important; font-size: 13px !important; }}
.missing {{ color: #cbd5e1; text-align: center; font-size: 24px; }}
.plot-card {{ display: block; width: 100%; border: 1px solid #e2e8f0; background: var(--soft); border-radius: 12px; padding: 7px; margin: 0 0 8px; cursor: zoom-in; text-align: left; transition: transform .12s, box-shadow .12s; }}
.plot-card:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px rgba(15,23,42,0.12); }}
.plot-card img {{ width: 100%; height: 145px; object-fit: contain; display: block; background: white; border-radius: 8px; }}
.plot-title {{ font-weight: 650; font-size: 12px; margin-bottom: 5px; line-height: 1.25; }}
.plot-file {{ color: var(--muted); font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 4px; }}
.empty {{ color: var(--muted); background: white; padding: 14px; border-radius: 12px; }}
.hidden {{ display: none !important; }}
#lightbox {{ position: fixed; inset: 0; z-index: 100; background: rgba(2,6,23,0.88); display: none; align-items: center; justify-content: center; padding: 24px; }}
#lightbox.open {{ display: flex; }}
.lightbox-panel {{ position: relative; width: min(96vw, 1500px); height: min(92vh, 1000px); background: white; border-radius: 16px; padding: 48px 16px 16px; box-shadow: 0 25px 80px rgba(0,0,0,0.45); }}
#lightboxTitle {{ position: absolute; left: 18px; top: 14px; right: 95px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.lightbox-close {{ position: absolute; right: 14px; top: 10px; border: 0; background: #0f172a; color: white; border-radius: 10px; padding: 8px 12px; cursor: pointer; }}
.zoom-scroll {{ width: 100%; height: 100%; overflow: auto; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0; display: flex; align-items: flex-start; justify-content: center; }}
#lightboxImg {{ max-width: none; transform-origin: top center; transition: transform .1s; padding: 12px; }}
.zoom-controls {{ position: absolute; right: 84px; top: 10px; display: flex; gap: 5px; }}
.zoom-controls button {{ border: 1px solid #cbd5e1; background: white; border-radius: 8px; padding: 6px 9px; cursor: pointer; }}
@media (max-width: 900px) {{ .matrix td {{ min-width: 180px; }} .plot-card img {{ height: 115px; }} }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class=\"subtitle\">Compact single-page benchmark dashboard. Click any plot to zoom; use filters to reduce the matrix.</div>
  <div class=\"stats\">
    <div class=\"stat\"><b>{total}</b>plots</div>
    <div class=\"stat\"><b>{ts}</b>time series</div>
    <div class=\"stat\"><b>{mg}</b>global maps</div>
    <div class=\"stat\"><b>{mc}</b>continent maps</div>
  </div>
</header>

<div class=\"controls\">
  <label>Experiment<select id=\"expFilter\"><option value=\"all\">All</option><option value=\"all-exp\">All-experiment time series only</option>{exp_options}</select></label>
  <label>Domain<select id=\"domainFilter\"><option value=\"all\">All</option>{domain_options}</select></label>
  <label>Metric<select id=\"metricFilter\"><option value=\"all\">All</option>{metric_options}</select></label>
  <label>Plot type<select id=\"kindFilter\"><option value=\"all\">All</option><option value=\"timeseries\">Time series</option><option value=\"map-global\">Global maps</option><option value=\"map-continent\">Continent maps</option></select></label>
  <label>Search<input id=\"searchBox\" placeholder=\"filename / expver / domain\"></label>
  <button class=\"small\" onclick=\"resetFilters()\">Reset</button>
</div>

<main>
  <section>
    <h2>Time series matrix</h2>
    <p class=\"note\">Rows are area domains; columns are benchmark metrics, differences and relative improvements. The standard time-series plot contains all experiments.</p>
    {ts_matrix}
  </section>

  <section>
    <h2>Global map matrix</h2>
    <p class=\"note\">Rows use the Global domain; columns are spatial metrics, difference maps and relative-improvement maps.</p>
    {global_matrix}
  </section>

  <section>
    <h2>Continental map matrix</h2>
    <p class=\"note\">Rows are continents; columns are the same map metrics at native-pixel resolution where available.</p>
    {continent_matrix}
  </section>
</main>

<div id=\"lightbox\" onclick=\"closeLightbox(event)\">
  <div class=\"lightbox-panel\" onclick=\"event.stopPropagation()\">
    <div id=\"lightboxTitle\"></div>
    <div class=\"zoom-controls\">
      <button onclick=\"zoomPlot(0.8)\">−</button>
      <button onclick=\"setZoom(1)\">100%</button>
      <button onclick=\"zoomPlot(1.25)\">+</button>
    </div>
    <button class=\"lightbox-close\" onclick=\"closeLightbox()\">Close</button>
    <div class=\"zoom-scroll\"><img id=\"lightboxImg\" src=\"\" alt=\"\"></div>
  </div>
</div>

<script>
let zoom = 1.0;
function matches(card, attr, value) {{
  if (value === 'all') return true;
  if (attr === 'expver' && value === 'all-exp') return card.dataset.expver === 'all';
  return card.dataset[attr] === value;
}}
function applyFilters() {{
  const exp = document.getElementById('expFilter').value;
  const dom = document.getElementById('domainFilter').value;
  const met = document.getElementById('metricFilter').value;
  const kind = document.getElementById('kindFilter').value;
  const q = document.getElementById('searchBox').value.toLowerCase();
  document.querySelectorAll('.plot-card').forEach(card => {{
    const txt = (card.innerText + ' ' + card.dataset.expver + ' ' + card.dataset.domain + ' ' + card.dataset.metric + ' ' + card.dataset.kind).toLowerCase();
    const ok = matches(card, 'expver', exp) && matches(card, 'domain', dom) && matches(card, 'metric', met) && matches(card, 'kind', kind) && txt.includes(q);
    card.classList.toggle('hidden', !ok);
  }});
}}
function resetFilters() {{
  for (const id of ['expFilter','domainFilter','metricFilter','kindFilter']) document.getElementById(id).value = 'all';
  document.getElementById('searchBox').value = '';
  applyFilters();
}}
for (const id of ['expFilter','domainFilter','metricFilter','kindFilter','searchBox']) {{
  document.addEventListener('DOMContentLoaded', () => document.getElementById(id).addEventListener('input', applyFilters));
}}
function openLightbox(btn) {{
  const data = JSON.parse(btn.dataset.plot);
  zoom = 1.0;
  const img = document.getElementById('lightboxImg');
  img.src = data.src;
  img.style.transform = 'scale(1)';
  document.getElementById('lightboxTitle').textContent = data.title + ' — ' + data.filename;
  document.getElementById('lightbox').classList.add('open');
}}
function closeLightbox(event) {{
  if (event && event.target.id !== 'lightbox') return;
  document.getElementById('lightbox').classList.remove('open');
  document.getElementById('lightboxImg').src = '';
}}
function setZoom(z) {{
  zoom = z;
  document.getElementById('lightboxImg').style.transform = `scale(${{zoom}})`;
}}
function zoomPlot(factor) {{
  zoom = Math.max(0.2, Math.min(6.0, zoom * factor));
  document.getElementById('lightboxImg').style.transform = `scale(${{zoom}})`;
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeLightbox(); }});
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact HTML dashboard for benchmark plots.")
    parser.add_argument("--outdir", required=True, help="Benchmark output directory")
    parser.add_argument("--title", default="CaMa-Flood vs GloFAS benchmark dashboard")
    parser.add_argument("--output", default=None, help="Output HTML file. Default: <outdir>/index.html")
    parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    if not outdir.exists():
        raise FileNotFoundError(outdir)

    html_path = Path(args.output).expanduser().resolve() if args.output else outdir / "index.html"
    items = scan_plots(outdir, html_path)
    if not items:
        print("WARNING: no PNG plots found under", outdir, file=sys.stderr)

    content = build_html(outdir, args.title, items, html_path)
    html_path.write_text(content, encoding="utf-8")
    print("Saved dashboard:", html_path)
    print("Plots indexed:", len(items))

    if args.open:
        webbrowser.open(html_path.as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
