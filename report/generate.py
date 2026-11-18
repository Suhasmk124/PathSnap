"""Build the consolidated PathSnap technical report from recorded outputs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]
INK="#15293a"
MUTED="#526778"
TEAL="#0c6259"
W,H=A4
CW=W-96

styles={
    "kicker":ParagraphStyle("kicker",fontName="Helvetica-Bold",fontSize=9,leading=14,textColor=colors.HexColor(TEAL),spaceAfter=8),
    "title":ParagraphStyle("title",fontName="Helvetica-Bold",fontSize=31,leading=35,textColor=colors.HexColor(INK),spaceAfter=13),
    "section":ParagraphStyle("section",fontName="Helvetica-Bold",fontSize=24,leading=29,textColor=colors.HexColor(INK),spaceAfter=11),
    "h":ParagraphStyle("h",fontName="Helvetica-Bold",fontSize=12,leading=16,textColor=colors.HexColor(INK),spaceBefore=9,spaceAfter=4),
    "body":ParagraphStyle("body",fontName="Helvetica",fontSize=10.4,leading=15.2,textColor=colors.HexColor(INK),spaceAfter=8),
    "caption":ParagraphStyle("caption",fontName="Helvetica",fontSize=8.7,leading=12,textColor=colors.HexColor(MUTED),spaceBefore=5,spaceAfter=8),
    "th":ParagraphStyle("th",fontName="Helvetica-Bold",fontSize=8.2,leading=11,textColor=colors.white),
    "td":ParagraphStyle("td",fontName="Helvetica",fontSize=8.7,leading=11.3,textColor=colors.HexColor(INK)),
}


def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percent(value):
    return f"{float(value):.2%}"


class Report:
    def __init__(self):
        self.story=[]

    def p(self,value,style="body"):
        self.story.append(Paragraph(value,styles[style]))

    def page(self,label,title):
        if self.story:
            self.story.append(PageBreak())
        self.p(label,"kicker")
        self.p(title,"section")

    def heading(self,title,body):
        self.p(title,"h")
        self.p(body)

    def table(self,rows,widths):
        cells=[[Paragraph(str(value),styles["th" if i==0 else "td"]) for value in row] for i,row in enumerate(rows)]
        table=Table(cells,colWidths=[CW*v for v in widths],hAlign="LEFT",repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(TEAL)),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#edf6f4"),colors.white]),
            ("LINEBELOW",(0,0),(-1,-1),.25,colors.HexColor("#dbe5e9")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ]))
        self.story.extend([table,Spacer(1,10)])

    def image(self,path,caption,max_h=340):
        width,height=ImageReader(str(path)).getSize()
        scale=min(CW/width,max_h/height)
        self.story.append(Image(str(path),width*scale,height*scale,hAlign="LEFT"))
        self.p(caption,"caption")


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor("#dbe5e9"))
    canvas.line(48,42,W-48,42)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.setFont("Helvetica",8)
    canvas.drawString(48,29,"PathSnap  /  Technical report")
    canvas.drawRightString(W-48,29,f"{doc.page}")


def build(path):
    results=ROOT/"results"
    trips=csv_rows(results/"p1/trip_inventory.csv")
    comparison=csv_rows(results/"p2/model_comparison.csv")
    diagnostics={r["model"]:r for r in csv_rows(results/"p2/fit_diagnostics.csv")}
    models={r["model"]:r for r in comparison}
    tuning=read_json(results/"p2/optuna/trials.json")
    hmm=read_json(results/"p3/student_test_summary.json")
    timings=csv_rows(results/"p4/direct_rehearsal.csv")
    names={"logistic":"Logistic regression","gaussian_nb":"Gaussian naive Bayes",
           "rbf_prototypes":"RBF prototypes","boosted_stumps":"Boosted stumps",
           "shallow_mlp":"Shallow MLP","shallow_mlp_tuned":"Shallow MLP (tuned)"}

    report=Report()
    report.p("MACHINE LEARNING HACKATHON  /  TECHNICAL REPORT","kicker")
    report.p("PathSnap","title")
    report.p("Distributed map matching and route reconstruction","section")
    report.p("PathSnap matches noisy, timestamped GPS measurements to directed road segments. A candidate classifier scores nearby arcs, and a Viterbi decoder reconstructs a sequence using the trip's road graph. The evaluator UI requests a trip from a separate API and renders the predicted route and server-computed metrics.")
    report.table([
        ["Student-test metric","Classifier only","Classifier + Viterbi"],
        ["Per-observation Top-1",percent(hmm["classifier"]["top1_accuracy"]),percent(hmm["hmm"]["top1_accuracy"])],
        ["Directed continuity using common time boundaries",percent(hmm["classifier"]["route_continuity"]),percent(hmm["hmm"]["route_continuity"])],
        ["Normalized arc-sequence edit distance (lower is better)",
         f"{hmm['classifier']['normalized_route_edit_distance']:.4f}",f"{hmm['hmm']['normalized_route_edit_distance']:.4f}"],
        ["Aligned GPS observations","19,160","19,160"],
    ],[.57,.19,.24])
    report.heading("Evaluation boundary","Model selection and tuning used the 80 training trips. The 10 student-test trips were used for the recorded model, decoder, and integration comparisons. Reserved evaluator trips were unavailable in the supplied bundle.")
    report.heading("Evidence included","The project repository contains trip inventory, radius coverage, fold assignments, measured model comparisons, all ten Bayesian trials, decoder tuning and per-trip metrics, interactive maps, and deployment source. The report figures are derived from those files.")
    report.heading("Submission evidence to complete","Generate the required Folium HTML on a machine with Folium available and capture a successful FastAPI response in the receiving laptop's terminal during the two-laptop rehearsal. The deployment page labels the evidence collected in this run.")
    report.p("AI assistance was used for implementation, experiment automation and report drafting. Results in this report come from the recorded project runs.")
    report.p("Source requirements: PathSnap Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, and Assessment and Plagiarism Policy.","caption")

    report.page("02  /  DATASET EXPLORATION","Dataset and preprocessing")
    rows=[["Partition","Trips","GPS points","Invalid GPS","Candidate miss"]]
    for split in ("train","student_test"):
        group=[r for r in trips if r["split"]==split]
        total=sum(int(r["gps_observations"]) for r in group)
        hit=sum(int(r["hits"]) for r in group)
        rows.append([split,len(group),f"{total:,}",sum(int(r["invalid_gps"]) for r in group),percent(1-hit/total)])
    report.table(rows,[.25,.12,.20,.18,.25])
    report.p("Each trip has a GPS track, node coordinates, directed arcs, and an ordered reference arc list. Longitude and latitude were converted to local metres using the median GPS coordinate. A monotone alignment inferred per-point reference arcs from the reference sequence using distance, heading, and an arc-skip penalty. These are inferred labels. The inventory also found zero missing GPS values and zero non-increasing timestamp steps in the supplied splits.")
    report.p("The inventory covers file sizes, coordinate extent, weak connectivity, isolated nodes, directionality, and speed and timestamp outliers. The largest road graph has 307,392 directed arcs. Training trip 90 includes a gap of about 42.5 hours. Training trips 54 and 33 account for roughly 79% of training candidate misses.")
    report.p("A training-only radius sweep selected 25 m and a 30-candidate cap. Features capture point-to-arc distance, projection fraction, heading agreement, speed, time gap, arc length, distance rank, and connectivity to the previous nearest candidate.")
    report.image(results/"p1/alignment_preview_00000075.png","Figure 1. Training example. GPS points, directed road geometry, reference route, sampled alignments, and two reference-route breaks.",260)

    report.page("03  /  MODEL EXPERIMENTATION","Classifier comparison")
    report.p("We used four trip-grouped folds. Median imputation and scaling were fitted inside each fold. Training used up to 4,500 sampled candidate rows per trip with inverse-square-root trip-size weights and a 1.5x multiplier for positives. Every validation and student-test row was scored. Candidate Macro F1 at threshold 0.5 was the selection objective.")
    columns=[["Model family","Key configuration","CV F1","Test Top-1","Continuity","Test F1","Log loss","Infer ms"]]
    cfg=["L2 .003, 100 steps","Variance floor .03","10 centers/class, bw .9",
         "60 rounds, rate .16","24 units, rate .006","41 units, rate .01997"]
    for row,setting in zip(comparison,cfg,strict=True):
        columns.append([names[row["model"]],setting,f"{float(row['cv_macro_f1']):.3f}",percent(row["top1_accuracy"]),
                        percent(row["route_continuity"]),f"{float(row['macro_f1']):.3f}",f"{float(row['log_loss']):.3f}",
                        f"{1000*float(row['inference_s']):.1f}"])
    report.table(columns,[.19,.21,.075,.11,.11,.075,.08,.12])
    report.image(results/"p2/plots/model_comparison.png","Figure 2. Recorded CV and student-test comparisons. The time column in the table measures classifier scoring across 97,676 student-test candidate rows.",250)
    report.p("The tuned MLP led grouped-CV Macro F1 and student-test Top-1. Gaussian naive Bayes had the highest classifier-only continuity at 80.48%, alongside the lowest Top-1 at 76.69%. The route decoder addresses sequence coherence separately.")

    report.page("03  /  MODEL EXPERIMENTATION","What each model learned")
    details={
        "logistic":"A linear weighted log-loss baseline tests whether candidate geometry separates classes additively. It scored quickly and had competitive Top-1, while the neural and boosted models achieved higher candidate Macro F1.",
        "gaussian_nb":"Weighted per-class Gaussian likelihoods give a fast fit. The features include related measurements such as distance rank and raw distance. Its 0.516 log loss and lower Top-1 make the independence and distribution assumptions a poor fit here.",
        "rbf_prototypes":"Weighted clustering gives prototypes for each class, followed by Gaussian-kernel density scoring. This improved log loss over logistic regression, while its lower Top-1 shows that better aggregate candidate probabilities did not improve the winning arc per point.",
        "boosted_stumps":"Sequential one-split trees reduce weighted logistic loss using binned features. This raised Macro F1 above the linear baseline. The additive stump architecture produced lower direct continuity and limits feature interactions.",
        "shallow_mlp":"A tanh hidden layer learns interactions between geometric and motion features. Adam optimized weighted binary log loss for seven epochs. It improved Macro F1 and Top-1 over boosted stumps.",
        "shallow_mlp_tuned":"The Bayesian search increased hidden width and learning rate. This fit had the highest grouped-CV Macro F1 and lowest held-out log loss, with a small increase in classifier scoring time.",
    }
    for key,description in details.items():
        d=diagnostics[key]
        row=models[key]
        report.heading(names[key],description+f" Student-test Top-1 {percent(row['top1_accuracy'])}, "
                       f"Macro F1 {float(row['macro_f1']):.3f}, log loss {float(row['log_loss']):.3f}. "
                       f"Fit-sample F1 exceeded grouped-CV F1 by {float(d['fit_minus_cv_f1']):.3f}.")
    report.p("All fit-sample gaps were modest relative to the cross-model spread. The fit subset and CV metrics cover different rows and trip mixes, so these diagnostics cannot rule out overfitting. No epoch-loss curves were recorded for this run.","caption")

    report.page("04  /  BAYESIAN OPTIMIZATION","Tuning the shallow MLP")
    best=tuning["trials"][tuning["best_trial"]]
    report.p("The search maximized mean grouped-CV candidate Macro F1. It sampled a learning rate between 0.001 and 0.020 and an integer hidden width from 12 to 44. The optimizer used a Gaussian-process surrogate with expected improvement after three initial random trials. The CPU run did not have Optuna available.")
    report.p(f"Trial {best['trial']+1} of {tuning['trial_count']} selected {best['params']['hidden']} hidden units and a learning rate of {best['params']['rate']:.5f}. Seven epochs were fixed. The recorded trial loop took {sum(r['elapsed_s'] for r in tuning['trials']):.1f} seconds across all folds and trials.")
    report.table([
        ["Metric","Untuned MLP","Tuned MLP"],
        ["Grouped CV candidate Macro F1","0.8568","0.8601"],
        ["Student-test Top-1","79.97%","80.27%"],
        ["Student-test candidate Macro F1","0.8753","0.8775"],
        ["Student-test log loss","0.1972","0.1887"],
        ["Student-test classifier scoring","19.6 ms","25.2 ms"],
    ],[.56,.22,.22])
    report.image(results/"p2/plots/bayesian_trials.png","Figure 3. All ten recorded trials and the fixed untuned reference.",248)

    report.page("05-06  /  DECODING AND GEOGRAPHY","Viterbi route reconstruction")
    report.p("The hidden states are directed candidate arcs. Emissions are the logarithms of clipped and per-point-normalized classifier scores. A transition compares observed GPS displacement with distance between two projected positions along the road network. Bounded Dijkstra follows directed arcs. The distance scale grows with the time interval and is relaxed for speed spikes.")
    report.p("Viterbi adds the current emission to the best predecessor's accumulated score plus its transition score, then follows backpointers. It starts a new piece after a gap over 30 seconds, an invalid clock step, a missing-candidate observation, or an unreachable transition. Stationary points penalize arc switches. A zero-candidate observation remains unassigned.")
    report.p("A 12-setting sweep chose a transition scale of 8 m and emission weight 1 using out-of-fold scores on 16 training trips. On student-test, Top-1 improved on all ten trips and by 12.85 points on trip 76. The decoder reduced the number of distinct transitions from 6,827 to 3,680 under identical time boundaries, so continuity rates are shown with their denominators.")
    report.image(ROOT/"maps/student_test_00000076_comparison.png","Figure 4. Student-test trip 76 with GPS, network, reference, and two predictions. The repo includes toggleable Leaflet HTML and a Folium-generation path.",276)
    report.p("Training stress case 54 went from 7.46% to 8.40% Top-1. Its 16,604 points without candidates explain why better route transitions alone cannot recover most of the trip.")

    report.page("07  /  DEPLOYMENT","API and evaluator UI")
    report.p("Laptop 1 runs FastAPI with the persisted MLP, the trip road network, candidate generation, Viterbi and server-side scoring. Laptop 2 serves the evaluator page and relays the trip-list and prediction requests. Only GPS coordinates, predicted arcs and aggregate metrics return to the evaluator UI.")
    report.p("The direct service rehearsal matched the offline P3 metrics on all ten student-test trips. Median direct elapsed time was "
             f"{median(float(r['elapsed_ms']) for r in timings):.0f} ms and the maximum was "
             f"{max(float(r['elapsed_ms']) for r in timings):.0f} ms. Timings include in-process inference and reference scoring and exclude HTTP.")
    report.image(ROOT/"report/ui_preview.png","Figure 5. Browser UI using the recorded trip-76 response from the prediction service. The browser supplied that response through an intercepted request during local testing.",300)
    report.p("For the 307,392-arc training stress case, inference took 1.96 s cold and 0.48 s with geometry, index and graph cached.")
    report.heading("Remaining demo evidence","FastAPI and Folium could not be installed in the local runtime and loopback HTTP was restricted. The two-laptop run, reserved trip list and receiving-laptop API terminal screenshot remain to be captured. The README contains the commands for that run.")
    report.p("Evidence files: results/p4/direct_rehearsal.csv, results/p3/student_test_metrics.csv, api/main.py, ui/app.py, and the experiment source under experiments/.","caption")

    path.parent.mkdir(parents=True,exist_ok=True)
    document=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=53,
                               title="PathSnap Technical Report",author="PathSnap")
    document.build(report.story,onFirstPage=footer,onLaterPages=footer)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    build(parser.parse_args().out)


