# Valida o gate :innovations regenerado contra owa_pareado.csv (referencia por serie do censo e do gate inov).
# Criterio: ordem (p,d,q,P,D,Q) igual e smape/mase iguais a 6 casas. Uso: python3 -I valida_inov_vs_owa.py <m1500_inov.csv> <owa_pareado.csv> <saida.csv>
import csv, sys
reg = {r['sid']: r for r in csv.DictReader(open(sys.argv[1]))}
ref = {r['sid']: r for r in csv.DictReader(open(sys.argv[2]))}
ordc = ['T','p','d','q','P','D','Q']
iguais = 0; div = []
with open(sys.argv[3], 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['sid','ordem_igual','smape_reg','smape_ref','mase_reg','mase_ref','igual_6casas','status_reg','iterMax_reg'])
    for sid, r in reg.items():
        o = ref.get(sid)
        if o is None: w.writerow([sid,'SEM_REF','','','','','False',r.get('status',''),r.get('iterMax','')]); div.append(sid); continue
        oi = all(str(int(float(r[c]))) == str(int(float(o[c]))) for c in ordc)
        sm = abs(float(r['smape']) - float(o['smape_J'])) < 5e-7
        ms = abs(float(r['mase']) - float(o['mase_J'])) < 5e-7
        ok = oi and sm and ms
        iguais += ok
        if not ok: div.append(sid)
        w.writerow([sid, oi, r['smape'], o['smape_J'], r['mase'], o['mase_J'], ok, r.get('status',''), r.get('iterMax','')])
print(f"series regeneradas: {len(reg)}  iguais a owa_pareado (ordem + smape/mase 6 casas): {iguais}  divergentes: {len(div)}")
print("divergentes:", ' '.join(div[:200]))
