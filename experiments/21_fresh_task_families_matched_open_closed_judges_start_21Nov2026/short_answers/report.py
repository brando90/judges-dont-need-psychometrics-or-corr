"""Generate the complete report from the frozen forecasts and measured loss tables."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def table(header, rows):
    lines = [header, ["---"]*len(header), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def span(row):
    return (f'{float(row["improvement"]):+.4f} [{float(row["lower_one_sided_975"]):+.4f}, '
            f'{float(row["upper_two_sided_95"]):+.4f}]')


def main():
    decisions = read("decisions.csv")
    scores = {(r["judge"],int(r["ratings_per_answer"]),int(r["training_ratings"]),r["method"],r["metric"]):
              float(r["mean"]) for r in read("scores.csv")}
    diffs = {(r["judge"],int(r["ratings_per_answer"]),int(r["training_ratings"]),r["comparator"],r["metric"]):
             r for r in read("contrasts.csv")}
    text = ["# When does cross-entropy calibration beat Pearson and uniform on new assignments? | 11-21-2026",
            "As of 11-21-2026. External human-rating study on University of North Texas computer-science short answers.",
            "Ten homework assignments contributed 160 answers to 67 questions and 320 individual correctness grades. "
            "Both judges received the same question, instructor answer, student answer and probability instructions. "
            "Each test fold held out an entire homework assignment. The primary condition used two human grades per training answer. "
            "For each fit, cross-entropy (CE) and Pearson searched the same 27 calibration transforms of a frozen judge. "
            "Each fold's estimate averages 200 seeded nested draws of training answers; every draw used identical labels for the two objectives. "
            "[Protocol and prompt](../PROTOCOL_11-21-2026.md) · [Corpus and graders](https://aclanthology.org/P11-1076/)",
            "## Which budgets cleared both comparisons?"]
    for judge, label in (("qwen","Qwen3.5-9B"),("claude","Claude Sonnet 5")):
        record = sorted((r for r in decisions if r["judge"]==judge and r["ratings_per_answer"]=="2"),
                        key=lambda r:int(r["training_ratings"]))
        success = [r["training_ratings"] for r in record if r["formal_status"]=="pass"]
        histogram = [r["training_ratings"] for r in record
                     if r["formal_status"]=="pass" and r["histogram_lower_positive"]=="1"]
        local_three = [r["training_ratings"] for r in record
                       if r["local_both_pass"]=="1" and r["histogram_lower_positive"]=="1"]
        local = [r["training_ratings"] for r in record if r["local_both_pass"]=="1"]
        counts = [int(r["training_ratings"]) for r in record if
                  all(float(diffs[judge,2,int(r["training_ratings"]),m,"rps"]["improvement"])>0
                      for m in ("pearson_fit","uniform"))]
        f = lambda a: ", ".join(map(str,a)) if a else "none"
        text.append(f"**{label}.** Budgets passing the descending confirmatory sequence: {f(success)}. "
                    f"Budgets with both local bounds above zero: {f(local)}. "
                    f"Budgets with both observed differences above zero: {f(counts)}. "
                    f"Among budgets passing the sequence, the CE fit also cleared the training-histogram comparison at: {f(histogram)}. "
                    f"Descriptively, all three local lower bounds exceeded zero at these budgets: {f(local_three)}.")
    text.extend(["Each comparison subtracts test ranked probability score (RPS) of the CE fit from the same test RPS of the named comparator. "
                 "Positive favors CE. Brackets below are nominal 95% two-sided overlap-adjusted t intervals over ten held-out homework assignments. "
                 "The same lower endpoint is the one-sided 97.5% bound used for each judge. Testing starts at 256 ratings and stops on the first budget "
                 "where either comparator’s lower bound fails to exceed zero. This fixed sequence limits searching across budgets, and each judge has one-sided alpha 0.025. "
                 "The t approximation with nine degrees of freedom can misestimate uncertainty with ten dependent folds. The 200 training draws measure label-sampling variability. "
                 "[One-sided p-values, all 1,024 fold-sign sensitivity permutations, and fold win counts](results/contrasts.csv) · "
                 "[Overlap correction source](https://doi.org/10.1023/A:1024068626366)",
                 "![Held-out ranked probability scores and paired improvements by exact training-rating budget](results/heldout_rps.png)",
                 "*Each fold withholds one homework. Both objectives exhaustively search 27 fixed transforms from the same smallest-distance-to-identity tie rule. "
                 "Normalized RPS ranges from zero to one, lower is better. Error bars concern paired differences on the same test answers.*"])
    for judge,label in (("qwen","Qwen3.5-9B, open weights"),("claude","Claude Sonnet 5, closed weights")):
        text.append("## " + label)
        grid=[]; panel=[]
        for b in (4,8,16,32,64,128,256):
            s=lambda m,x:scores[judge,2,b,m,x]
            d=lambda m:diffs[judge,2,b,m,"rps"]
            grid.append([b,*[f'{s(m,"rps"):.4f}' for m in ("ce_fit","pearson_fit","uniform","train_hist","identity")],
                         span(d("pearson_fit")),span(d("uniform"))])
            panel.append([b,*[f'{s(m,x):.4f}' for x in ("ce","tvd") for m in ("ce_fit","pearson_fit","uniform","train_hist","identity")]])
        text.append(table(["Training ratings","CE fit RPS","Pearson fit RPS","Uniform RPS","Histogram RPS","Original RPS",
                           "Pearson minus CE fit [95%]","Uniform minus CE fit [95%]"],grid))
        text.append("Test CE is the negative natural log probability of an individual grade, in nats. "
                    "Test total variation distance (TVD) is half the absolute probability differences summed across grades against each answer's two-grader histogram. "
                    "Both are lower-is-better. TVD is noisy with two ratings per answer. The histogram uses training ratings only, with a half-count per grade.")
        text.append(table(["Training ratings",*[m+" test "+v for v in ("CE","TVD") for m in ("CE fit","Pearson fit","Uniform","Histogram","Original")]],panel))
    text.extend(["## What happens with one rating per training answer?",
                 "This secondary condition uses twice as many distinct answers as the primary condition at a matched total rating budget. "
                 "It uses the same held-out answers with both graders' outcomes. The table shows descriptive paired RPS differences."])
    rows=[]
    for b in (4,8,16,32,64,128):
        rows.append([b,*[span(diffs[j,1,b,m,"rps"]) for j in ("qwen","claude") for m in ("pearson_fit","uniform")]])
    text.append(table(["Training ratings","Qwen Pearson minus CE fit","Qwen uniform minus CE fit","Claude Pearson minus CE fit","Claude uniform minus CE fit"],rows))
    with (ROOT/"forecasts.csv").open() as f:
        forecasts=list(csv.DictReader(f))
    accounting=[]
    for judge in ("qwen","claude"):
        group=[r for r in forecasts if r["judge"]==judge]
        accounting.append([judge,len(group),*[sum(r["status"]==key for r in group) for key in ("valid","invalid_uniform","failure_uniform")]])
    text.extend(["## Coverage and interpretation",table(["Judge","Requests","Valid","Invalid output → uniform","Failed call → uniform"],accounting),
                 "The prespecified fallback retains all 160 test answers per judge. A calibrated fallback can use a training-set rating prior. "
                 "Openness, provider, model architecture and decoding differ together. The two fixed graders and one course limit generalization. Student identities are absent from the released data, so the same students may contribute to training and test homework. "
                 "The corpus was public before these models were released; model pretraining exposure is unknown. "
                 "No new VeriBench human outcome enters these estimates. [Absolute metrics and within-fold training-draw variation](results/scores.csv) · "
                 "[Per-family losses](results/families.csv) · [Held-out Pearson, Spearman, and absolute agreement](results/diagnostics.csv) · "
                 "[Paired calibration settings and the reason for convergence](HEAD_AGREEMENT_11-21-2026.md) · [Selection counts](results/selections.csv) · [Undefined training correlations](results/undefined_pearson.csv)"])
    (ROOT / "RESULTS_11-21-2026.md").write_text("\n\n".join(text)+"\n")


if __name__ == "__main__":
    main()
