"""Print JSON values behind the numbers quoted in RESULTS/TLDR (efdc8c3) for side-by-side checking."""
import json, sys
J = json.load(open(sys.argv[1]))
def pr(path, blk, keys=("RPS","CE","TVD","Brier","ICC_A1","Bias","Mean_Judge","Mean_Human")):
    p = blk["point"]; print(path, {k: round(p[k],4) for k in keys if k in p})
def dd(path, d, k="dRPS"):
    x = d[k]; print("  ", path, k, round(x["point"],4), [round(v,4) for v in x["ci95"]], "p=", x["p_val_two_sided"], "n_ge0=", x.get("n_ge_zero"))
e = J["exp27_split_test"]; c = J["cv5_nested_oof"]
for k in ("constant_set_aware","exp27_f3_h3free","exp27_f8_refit250","exp27_f9_selected","exp27_f9_f8hp","winner_global","winner_set_aware","c9b_zero_opus_fallback","frontier_global_selected","frontier_set_c7_std3"):
    pr("exp27."+k, e[k]); dd("vs_const", e[k]["vs_const_set"]); dd("vs_f8", e[k]["vs_exp27_f8"]); dd("vs_f3", e[k]["vs_exp27_f3_h3free"])
dd("winner_global seed24 vs const", e["winner_global_seed20261124"]["vs_const_set"])
dd("C7 - F9sel", e, "winner_set_aware_vs_f9_selected") if False else None
print("  C7-F9sel", {k: round(v,4) if isinstance(v,float) else v for k,v in e["winner_set_aware_vs_f9_selected"]["dRPS"].items() if k in ("point","p_val_two_sided")}, e["winner_set_aware_vs_f9_selected"]["dRPS"]["ci95"])
print("  C7-F9f8hp", e["winner_set_aware_vs_f9_f8hp"]["dRPS"]["point"], e["winner_set_aware_vs_f9_f8hp"]["dRPS"]["ci95"], e["winner_set_aware_vs_f9_f8hp"]["dRPS"]["p_val_two_sided"])
for k in ("constant_set_aware","exp27_f3_h3free","exp27_f8_h3_exposed","nested_winner_global","nested_frontier_global","nested_winner_set_aware","nested_frontier_set_aware","exploratory_fixed_hp_winner_global","exploratory_fixed_hp_winner_set_aware"):
    pr("cv5."+k, c[k]); dd("vs_const", c[k]["vs_const_set"]); dd("vs_f8", c[k]["vs_exp27_f8"]); dd("vs_f3", c[k]["vs_exp27_f3_h3free"])
for lab, h in (("exp27_head", J["exp27_head_matched"]), ("cv5_c7", J["head_matched_cv5_c7"]), ("cv5_g9", J["head_matched_cv5_g9"])):
    for o in ("ce","rps","pearson"): pr(lab+"."+o, h[o])
    dd(lab+" CE-Pear", h["ce_vs_pearson"]); dd(lab+" CE-Pear", h["ce_vs_pearson"], "dCE"); dd(lab+" CE-Pear", h["ce_vs_pearson"], "dTVD"); dd(lab+" RPS-Pear", h["rps_vs_pearson"])
for sp in ("exp27_test","cv5_oof"):
    L = J["qwen3_4b_lora"][sp]; dd("lora "+sp+" CE-Pear", L["ce_vs_pearson"]); dd("lora "+sp+" CE-Pear", L["ce_vs_pearson"], "dCE"); dd("lora "+sp+" RPS-Pear", L["rps_vs_pearson"])
print("nested selections (global, frontier_global):")
for s in J["nested_fold_selections"]:
    print("  fold", s["fold"], s["global"]["candidate"], s["global"]["objective"], s["global"]["alpha"], "|", s["frontier_global"]["candidate"], s["frontier_global"]["objective"], s["frontier_global"]["alpha"], "| set:", s["set_aware_h3free"]["candidate"])
print("pearson head HPs by fold c7:", [(h["pearson"]["alpha"], h["pearson"]["dirichlet_eta"], h["pearson"]["use_quad"]) for h in J["head_matched_cv5_c7"]["selected_hps_by_fold"]])
print("ce head HPs by fold c7:", [(h["ce"]["alpha"], h["ce"]["dirichlet_eta"], h["ce"]["use_quad"]) for h in J["head_matched_cv5_c7"]["selected_hps_by_fold"]])
print("exp27 head HPs:", {o: (J["exp27_head_matched"]["selected_hps"][o]["alpha"], J["exp27_head_matched"]["selected_hps"][o]["dirichlet_eta"]) for o in ("ce","rps","pearson")})
