import csv, sys, statistics as st
o = list(csv.DictReader(open(sys.argv[1]))); n = list(csv.DictReader(open(sys.argv[2])))
def dsel(r): return r["order"].strip("()").split(",")[1]
print("subset,code,cases,true_order_selected,d_selected_equals_true,median_rmse12,mean_rmse12")
for label, filt in (("semantics = DGP", lambda r: r["dgp"] == r["semantics"]),
                    ("semantics = DGP and d selected = d of the DGP", lambda r: r["dgp"] == r["semantics"] and dsel(r) == r["d_true"]),
                    ("all cases", lambda r: True)):
    for code, rows in (("v1.0.0", o), ("fixed", n)):
        sub = [r for r in rows if filt(r)]
        if label.startswith("semantics = DGP and"):
            # same case set for both codes: the d test runs before the search and is unchanged
            pass
        true = sum(r["order"] == f"(1,{r['d_true']},1)" for r in sub)
        dok = sum(dsel(r) == r["d_true"] for r in sub)
        rm = [float(r["rmse12"]) for r in sub]
        print(f"{label},{code},{len(sub)},{true},{dok},{st.median(rm):.4f},{st.mean(rm):.4f}")
