"""Generate a PDF-ready offline comparison from the recorded student-test predictions."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .maps import layers
from .p3 import records
from src.geo import Trip, project

ROOT = Path(__file__).resolve().parents[2]


def render(trip_dir, rows, metrics, destination):
    trip=Trip.load(trip_dir)
    gps, all_lines=layers(trip,rows,trip_dir)
    tracks=np.asarray(gps)[:,::-1]
    origin=np.median(tracks,axis=0)
    track=project(tracks,origin)
    segments=[project(np.asarray(lines).reshape(-1,2)[:,::-1],origin).reshape(-1,2,2) for lines in all_lines]
    lower, upper=track.min(axis=0),track.max(axis=0)
    mid=(lower+upper)/2
    span=np.maximum(upper-lower,1)
    scale=min(550/span[0],490/span[1])
    image=Image.new("RGB",(1420,748),"#f4f7fb")
    drawing=ImageDraw.Draw(image)

    def label(x,y,value,size,color="#172745"):
        drawing.text((x,y),value,fill=color,font=ImageFont.load_default(size=size))

    label(30,24,"PathSnap | Classifier and Viterbi route",33)
    label(30,68,f"Student-test trip {trip.trip_id}  ·  GPS points, road network and reference",18)
    colors=("#b0bdce","#188242","#be185d","#f06b20")
    for panel,(name,k) in enumerate((("Classifier only",2),("Classifier + Viterbi",3))):
        x=30+panel*695
        label(x,111,f"{name}   |   Top-1 {metrics[panel]:.2%}",21)
        plot=Image.new("RGB",(665,548),"white")
        d=ImageDraw.Draw(plot)

        def pixels(xy):
            return ((xy-mid)*[scale,-scale]+[332,274]).round().astype(int)

        for index in (0,1,k):
            for a,b in pixels(segments[index]):
                d.line((tuple(a),tuple(b)),fill=colors[index],width=(1,6,3)[(0,1,k).index(index)])
        for px,py in pixels(track):
            d.ellipse((px-1,py-1,px+1,py+1),fill="#2563eb")
        image.paste(plot,(x,146))
        drawing.rectangle((x,146,x+664,693),outline="#d4deed")
    x=30
    for name,color in zip(("Road network","Reference route","Classifier route","Viterbi route","GPS observations"),
                          (*colors,"#2563eb")):
        drawing.rectangle((x,716,x+15,729),fill=color)
        label(x+23,711,name,17)
        x+=222
    image.save(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--out",type=Path,default=ROOT/"maps")
    args=parser.parse_args()
    metrics={}
    for row in records(args.results/"student_test_metrics.csv"):
        metrics.setdefault(row["trip_id"],{})[row["method"]]=float(row["top1_accuracy"])
    trip=max(metrics,key=lambda k:metrics[k]["hmm"]-metrics[k]["classifier"])
    rows=list(records(args.results/"points"/f"student_test_{trip}.csv.gz"))
    args.out.mkdir(parents=True,exist_ok=True)
    render(args.raw/"student_test"/trip,rows,(metrics[trip]["classifier"],metrics[trip]["hmm"]),
           args.out/f"student_test_{trip}_comparison.png")


if __name__=="__main__":
    main()


