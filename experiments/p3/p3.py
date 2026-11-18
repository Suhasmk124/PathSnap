"""Tune on training OOF scores and evaluate on the 10 student-test trips."""
import argparse
from collections import defaultdict
import csv
import gzip
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src._p1_data import load_reference
from src.eval.metrics import observation_metrics
from src.geo import Geometry, Trip
from src.hmm import Scores, load_config, prepare, viterbi

ROOT = Path(__file__).resolve().parents[2]
MODEL = "shallow_mlp_tuned"
STRESS = ("00000090", "00000078", "00000032", "00000079", "00000022", "00000013", "00000054")


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def get_scores(path, trip_ids):
    selected = defaultdict(list)
    for row in records(path):
        trip = row["trip_id"]
        if trip in trip_ids:
            selected[trip].append((int(row["obs_idx"]), int(row["arc_id"]), float(row["score"])))
    result = {}
    for trip in trip_ids:
        rows = np.array(selected[trip], dtype=float).reshape((-1,3))
        result[trip] = Scores(rows[:,0].astype(np.int32), rows[:,1].astype(np.int32), rows[:,2])
    return result


def point_references(path):
    out = defaultdict(list)
    for row in records(path):
        out[row["split"], row["trip_id"]].append(
            (int(row["observation_index"]), int(row["reference_arc_id"]), not int(row["candidate_miss"])))
    return out


def point_arrays(n, reference):
    data = np.asarray(reference, dtype=np.int32)
    truth = np.full(n,-1,dtype=np.int32)
    present = np.zeros(n,dtype=bool)
    truth[data[:,0]] = data[:,1]
    present[data[:,0]] = data[:,2]
    return truth, present


def downsample_by_size(ref):
    candidates = sorted((len(points), trip) for (split,trip),points in ref.items() if split=="train")
    return [candidates[i][1] for i in np.linspace(0,len(candidates)-1,16,dtype=int)]


def sequence(arcs):
    ids = arcs[arcs>=0]
    return ids[np.r_[True,np.diff(ids)!=0]] if len(ids) else ids


def edit_distance(reference, prediction):
    # Bit-vector Levenshtein. Each bit represents a position in reference.
    if not len(reference):
        return len(prediction)
    masks = defaultdict(int)
    for i, arc in enumerate(reference):
        masks[int(arc)] |= 1 << i
    positive, negative = ~0, 0
    distance = len(reference)
    top = 1 << (distance-1)
    for arc in prediction:
        matches = masks[int(arc)]
        xv = matches | negative
        xh = (((matches & positive) + positive) ^ positive) | matches
        ph = negative | ~(xh | positive)
        mh = positive & xh
        distance += bool(ph & top) - bool(mh & top)
        ph = (ph << 1) | 1
        mh <<= 1
        positive = mh | ~(xv | ph)
        negative = ph & xv
    return distance


def trajectory_metric(g, truth, in_cands, pred, piece):
    use = pred>=0
    source = np.full(len(pred),-1,dtype=np.int32)
    target = source.copy()
    source[use], target[use] = g.trip.arcs[pred[use]].T
    return observation_metrics(truth,in_cands,pred,source,target,piece)


def input_pieces(g, gap):
    return np.cumsum(~np.isfinite(g.dt) | (g.dt<=0) | (g.dt>gap))


def tune(raw, score_path, reference, default, output):
    trips = downsample_by_size(reference)
    scores = get_scores(score_path, set(trips))
    trials = [({**default, "beta_m": beta, "emission_weight": weight}, [])
              for beta,weight in product((8.,16.,32.,64.), (1.,2.,4.))]
    for trip in trips:
        g = Geometry(Trip.load(raw/"train"/trip))
        truth,present = point_arrays(len(g.gps),reference["train",trip])
        pieces = input_pieces(g,default["gap_threshold_s"])
        prepared = prepare(g,scores[trip],default)
        for config, values in trials:
            _,pred,_,_ = viterbi(prepared,config)
            metrics = trajectory_metric(g,truth,present,pred,pieces)
            values.append((metrics["top1_accuracy"],metrics["route_continuity"]))
    logs = []
    for config, values in trials:
        logs.append(dict(beta_m=config["beta_m"],emission_weight=config["emission_weight"],trips=len(trips),
                         train_macro_top1=float(np.mean([a for a,_ in values])),
                         train_macro_continuity=float(np.mean([b for _,b in values if b is not None]))))
    winner = max(range(len(trials)),key=lambda j:(logs[j]["train_macro_top1"],logs[j]["train_macro_continuity"]))
    write_csv(output/"training_decoder_sweep.csv",logs)
    (output/"tuning_trip_ids.json").write_text(json.dumps(trips,indent=2),encoding="utf-8")
    print("Training-only decoder selection:", logs[winner],flush=True)
    return trials[winner][0]


def evaluate(raw, split, score_path, reference, trips, config, output):
    scores = get_scores(score_path,set(trips))
    rows = []
    for trip in trips:
        folder=raw/split/trip
        start=perf_counter()
        g=Geometry(Trip.load(folder))
        prepared=prepare(g,scores[trip],config)
        clf,hmm,pieces,breaks=viterbi(prepared,config)
        elapsed=perf_counter()-start
        truth,present=point_arrays(len(g.gps),reference[split,trip])
        raw_route=load_reference(folder,len(g.length))
        route=sequence(raw_route)
        natural=input_pieces(g,config["gap_threshold_s"])
        for name,pred in (("classifier",clf),("hmm",hmm)):
            shared=trajectory_metric(g,truth,present,pred,natural)
            contract=trajectory_metric(g,truth,present,pred,pieces)
            arc_seq=sequence(pred)
            edits=edit_distance(route,arc_seq)
            denominator=max(len(route),len(arc_seq),1)
            rows.append(dict(split=split,trip_id=trip,method=name,**shared,
                             decoder_piece_continuity=contract["route_continuity"],
                             decoder_piece_transitions=contract["continuity_transitions"],
                             route_edit_distance=edits,normalized_route_edit_distance=edits/denominator,
                             reference_arc_count=len(route),predicted_arc_count=len(arc_seq),
                             decoder_pieces=int(pieces.max())+1,load_and_decode_s=elapsed,
                             **breaks))
        with gzip.open(output/"points"/f"{split}_{trip}.csv.gz","wt",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(("trip_id","obs_idx","t","pred_arc_clf","pred_arc_hmm","piece_id"))
            writer.writerows((trip,i,t,a,b,p) for i,(t,a,b,p) in enumerate(zip(g.trip.track[:,2],clf,hmm,pieces)))
        print(f"{split} {trip}: {rows[-2]['top1_accuracy']:.3%} -> {rows[-1]['top1_accuracy']:.3%}; {elapsed:.2f}s",
              flush=True)
    return rows


def summary(rows):
    results={}
    for method in ("classifier","hmm"):
        ours=[r for r in rows if r["method"]==method]
        def weighted(key, weight):
            valid=[r for r in ours if r[key] is not None]
            return sum(r[key]*r[weight] for r in valid)/sum(r[weight] for r in valid)
        results[method]=dict(
            trips=len(ours),aligned_points=sum(r["aligned_points"] for r in ours),
            top1_accuracy=weighted("top1_accuracy","aligned_points"),
            candidate_miss_rate=weighted("candidate_miss_rate","aligned_points"),
            route_continuity=weighted("route_continuity","continuity_transitions"),
            continuity_transitions=sum(r["continuity_transitions"] for r in ours),
            decoder_piece_continuity=weighted("decoder_piece_continuity","decoder_piece_transitions"),
            decoder_piece_transitions=sum(r["decoder_piece_transitions"] for r in ours),
            normalized_route_edit_distance=float(np.mean([r["normalized_route_edit_distance"] for r in ours])),
        )
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--p1",type=Path,required=True)
    parser.add_argument("--p2",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"results")
    parser.add_argument("--model",default=MODEL)
    args=parser.parse_args()
    out=args.out
    (out/"points").mkdir(parents=True,exist_ok=True)
    cfg=load_config()
    refs=point_references(args.p1/"point_references.csv")
    train_scores=args.p2/f"{args.model}_train.csv.gz"
    test_scores=args.p2/f"{args.model}_student_test.csv.gz"
    selected=tune(args.raw,train_scores,refs,cfg,out)
    selected["model_name"]=args.model
    selected["selection_metric"]="macro per-trip Top-1 on 16 training trips using OOF scores"
    (out/"selected_config.yaml").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    test=evaluate(args.raw,"student_test",test_scores,refs,
                  sorted(trip for split,trip in refs if split=="student_test"),selected,out)
    stress=evaluate(args.raw,"train",train_scores,refs,STRESS,selected,out)
    write_csv(out/"student_test_metrics.csv",test)
    write_csv(out/"stress_test_metrics.csv",stress)
    (out/"student_test_summary.json").write_text(json.dumps(summary(test),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

