"""Offline report maps for student-test classifier and Viterbi results."""
import argparse
import json
from pathlib import Path

import numpy as np

from .p3 import records
from src._p1_data import load_reference
from src.geo import Trip

ROOT = Path(__file__).resolve().parents[2]

LAYERS = [
    ("Road network", "#9aa8bb", 1, False),
    ("Reference route", "#15803d", 6, True),
    ("Classifier route", "#be185d", 3, False),
    ("Viterbi route", "#f97316", 3, True),
]


def layers(trip, rows, folder):
    track = trip.track[:, :2]
    gps = track[np.isfinite(track).all(axis=1), ::-1].round(6).tolist()
    bounds = np.asarray(gps)[:, ::-1]
    lo, hi = bounds.min(axis=0) - .0015, bounds.max(axis=0) + .0015
    coords = trip.nodes[trip.arcs]
    local = np.flatnonzero(((coords >= lo) & (coords <= hi)).all(axis=2).any(axis=1))
    if len(local) > 14000:
        local = local[np.linspace(0, len(local)-1, 14000, dtype=int)]
    labels = load_reference(folder,len(trip.arcs))
    clf = np.array([int(row["pred_arc_clf"]) for row in rows])
    hmm = np.array([int(row["pred_arc_hmm"]) for row in rows])
    def to_lines(ids):
        return coords[np.unique(ids[ids>=0]), :, ::-1].round(6).tolist()
    return gps,[to_lines(ids) for ids in (local, labels, clf, hmm)]


def generate(path, trip_dir, rows):
    trip = Trip.load(trip_dir)
    gps,lines = layers(trip,rows,trip_dir)
    try:
        import folium
    except ModuleNotFoundError as error:
        if error.name != "folium":
            raise
        leaflet(path,trip.trip_id,gps,lines)
        return "Leaflet"
    view = folium.Map(prefer_canvas=True)
    for (name,color,width,shown), coords in zip(LAYERS,lines):
        layer = folium.FeatureGroup(name=name,show=shown).add_to(view)
        if coords:
            folium.PolyLine(coords,color=color,weight=width,opacity=.82).add_to(layer)
    points=folium.FeatureGroup(name="GPS observations",show=True).add_to(view)
    for point in gps:
        folium.CircleMarker(point,radius=2,color="#2563eb",fill=True).add_to(points)
    view.fit_bounds(gps)
    folium.LayerControl(collapsed=False).add_to(view)
    legend=("<b>PathSnap " + trip.trip_id + "</b><br>" +
            "<br>".join(f"<span style='color:{color}'>&#9632;</span> {name}" for name,color,_,_ in LAYERS)
            + "<br><span style='color:#2563eb'>&#9679;</span> GPS observations")
    view.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:26px;left:18px;z-index:9999;background:#fff;"
        "color:#172b45;padding:10px 13px;line-height:1.6;border-radius:5px'>" + legend + "</div>"))
    view.save(path)
    return "Folium"


def leaflet(path, trip, gps, lines):
    template = """<!doctype html>
<html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathSnap route comparison — __TRIP__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body{margin:0;font:15px system-ui;color:#162747}header{padding:10px 16px}
#map{height:calc(100dvh - 48px)}.legend{padding:9px 12px;background:white;line-height:1.65;border-radius:6px}
</style>
<header><strong>Trip __TRIP__</strong> · Classifier and Viterbi route comparison</header>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const payload=__PAYLOAD__;
const map=L.map('map',{preferCanvas:true}).fitBounds(payload.gps);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {attribution:'&copy; OpenStreetMap contributors',maxZoom:19}).addTo(map);
const overlays={};
payload.layers.forEach(([name,color,weight,show],i)=>{
  const layer=L.polyline(payload.lines[i],{color,weight,opacity:.83});
  overlays[name]=layer;if(show)layer.addTo(map)
});
const points=L.layerGroup(payload.gps.map(p=>L.circleMarker(p,{radius:2,color:'#2563eb',fillOpacity:.7})));
overlays['GPS observations']=points;points.addTo(map);
L.control.layers({},overlays,{collapsed:false}).addTo(map);
const legend=L.control({position:'bottomleft'});
legend.onAdd=()=>{const box=L.DomUtil.create('div','legend');
  [...payload.layers,['GPS observations','#2563eb']].forEach(([n,c])=>{
    const item=document.createElement('div');const color=document.createElement('span');
    color.style.color=c;color.textContent='■ ';item.append(color,n);box.append(item);
  });return box;};
legend.addTo(map);
</script></html>"""
    data = json.dumps(dict(gps=gps,lines=lines,layers=LAYERS),separators=(",",":"))
    path.write_text(template.replace("__TRIP__",trip).replace("__PAYLOAD__",data),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--out",type=Path,default=ROOT/"maps")
    args=parser.parse_args()
    by_trip={}
    for row in records(args.results/"student_test_metrics.csv"):
        by_trip.setdefault(row["trip_id"],{})[row["method"]]=float(row["top1_accuracy"])
    ranked=sorted(by_trip,key=lambda trip:by_trip[trip]["hmm"]-by_trip[trip]["classifier"])
    args.out.mkdir(parents=True,exist_ok=True)
    for trip,role in ((ranked[-1],"largest_gain"),(ranked[0],"smallest_gain")):
        rows=list(records(args.results/"points"/f"student_test_{trip}.csv.gz"))
        path=args.out/f"student_test_{trip}_{role}.html"
        renderer=generate(path,args.raw/"student_test"/trip,rows)
        print(f"{renderer}: {path.name}")


if __name__=="__main__":
    main()


