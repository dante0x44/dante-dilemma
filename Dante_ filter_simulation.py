#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   DANTE FILTER — MONTE CARLO SIMULATION v2.0                        ║
║   Autor: 0x44 DANTE (Benjamin Limanovska) | Malta 2026              ║
╚══════════════════════════════════════════════════════════════════════╝
pip install numpy matplotlib
"""
import numpy as np, tkinter as tk, threading, time, json, csv, os, sys, pickle
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('TkAgg')

# ── Konstanten ────────────────────────────────────────────────────────
OUTCOME_NAMES  = ["Totaler Kollaps","Digitale Dystopie","Freiwilliger Verzicht","Evolutionärer Sprung"]
OUTCOME_COLORS = ["#FF2020","#FF8C00","#FFD700","#00FF7F"]
ACTOR_TYPES    = {"Aggressiv":{"trust":-0.20,"risk":+0.25,"coop":-0.20},
                  "Vorsichtig":{"trust":+0.10,"risk":-0.20,"coop":+0.10},
                  "Kooperativ":{"trust":+0.20,"risk":-0.15,"coop":+0.25},
                  "Gemischt":  {"trust": 0.00,"risk": 0.00,"coop": 0.00}}
BG,BG2,FG,ACC,BRD = '#0a0a0f','#0d0d1a','#aaaacc','#FF4500','#1a1a3a'
BATCH = 400_000

# ── SIMULATION CORE ───────────────────────────────────────────────────

def _apply_actor_type(base_val, delta, n):
    return np.clip(np.random.normal(base_val + delta, 0.12, n), 0, 1)

def simulate_batch(n, escalation, n_actors, trust_pct, info_pct,
                   risk_pct, coop_pct, actor_type="Gemischt",
                   n_rounds=1, shock_prob=0.0, shock_strength=0.3):
    """
    Vollständige Monte-Carlo-Simulation mit:
    - Akteur-Typen (Aggressiv/Vorsichtig/Kooperativ)
    - Mehreren Runden (dynamische Zeitentwicklung)
    - Externen Schocks
    - Netzwerk-Effekten (Misstrauen überträgt sich)
    """
    dt = ACTOR_TYPES[actor_type]
    tb = trust_pct/100 + dt["trust"]
    rb = risk_pct /100 + dt["risk"]
    cb = coop_pct /100 + dt["coop"]
    ib = info_pct  /100

    paranoia    = (escalation/10.0)**1.7
    t_pressure  = (8.0 + (escalation-1)*7.11)/72.0

    # State-Arrays (n_sims × n_actors)
    trust = np.clip(np.random.normal(tb, 0.12,(n,n_actors)),0,1)
    info  = np.clip(np.random.normal(ib, 0.08,(n,n_actors)),0,1)
    risk  = np.clip(np.random.normal(rb, 0.10,(n,n_actors)),0,1)
    coop  = np.clip(np.random.normal(cb, 0.10,(n,n_actors)),0,1)

    final_builders = np.zeros(n, dtype=np.float32)

    for rnd in range(max(1, n_rounds)):
        # Externe Schocks: zufällig Vertrauen senken
        if shock_prob > 0:
            shocks = np.random.random((n, n_actors)) < shock_prob
            trust  = np.clip(trust - shocks * shock_strength, 0, 1)
            coop   = np.clip(coop  - shocks * shock_strength * 0.5, 0, 1)

        threat   = info * (1-trust) * paranoia
        nash_p   = 1.0 - np.power(np.clip(trust*coop,0,1), 2.2)
        p_build  = np.clip(0.35*threat + 0.45*nash_p*t_pressure + 0.20*(1-coop)*risk, 0,1)
        decisions= np.random.random((n,n_actors)) < p_build
        n_b      = decisions.sum(axis=1)
        final_builders += n_b.astype(np.float32)

        # Netzwerk-Effekt: wenn Nachbar eskaliert → Misstrauen steigt
        if n_rounds > 1 and rnd < n_rounds-1:
            frac_esc = n_b / n_actors
            trust   -= frac_esc[:,None] * 0.15
            trust    = np.clip(trust, 0, 1)
            coop    -= frac_esc[:,None] * 0.10
            coop     = np.clip(coop, 0, 1)

    avg_b     = final_builders / n_rounds
    threshold = n_actors * 0.5

    collapse  = avg_b > threshold
    dystopia  = (~collapse) & (avg_b >= 1)
    no_build  = avg_b < 1

    avg_trust = trust.mean(axis=1)
    avg_coop  = coop.mean(axis=1)
    leap      = no_build & (avg_trust > 0.88) & (avg_coop > 0.80)
    restraint = no_build & ~leap

    outcomes = np.zeros(n, dtype=np.int8)
    outcomes[collapse]  = 0
    outcomes[dystopia]  = 1
    outcomes[restraint] = 2
    outcomes[leap]      = 3

    stats = dict(
        avg_trust     = float(avg_trust.mean()),
        avg_coop      = float(avg_coop.mean()),
        avg_builders  = float(avg_b.mean()),
        collapse_rate = float(collapse.mean()),
    )
    return np.bincount(outcomes, minlength=4), stats


def sensitivity_scan(param_a_vals, param_b_vals, fixed: dict,
                     n_sims=50_000, param_a='trust_pct', param_b='coop_pct',
                     callback=None):
    """2D-Sensitivitätsanalyse → Heatmap-Matrix"""
    rows, cols = len(param_a_vals), len(param_b_vals)
    heatmap = np.zeros((rows, cols))
    total   = rows * cols
    done    = [0]
    for i,a in enumerate(param_a_vals):
        for j,b in enumerate(param_b_vals):
            kw = dict(fixed)
            kw[param_a] = a
            kw[param_b] = b
            counts,_ = simulate_batch(n_sims, **kw)
            heatmap[i,j] = 100.0 * counts[0] / max(counts.sum(),1)
            done[0] += 1
            if callback:
                callback(done[0], total, heatmap)
    return heatmap


# ── GUI ───────────────────────────────────────────────────────────────

class DanteV2:
    def __init__(self, root):
        self.root = root
        root.title("DANTE FILTER v2.0 — 0x44 DANTE")
        root.configure(bg=BG)
        root.geometry("1600x960")

        # State
        self.running  = False
        self.results  = np.zeros(4, dtype=np.int64)
        self.done     = 0
        self.target   = 0
        self.t0       = None
        self.hist_collapse = []
        self.hist_x        = []
        self.all_stats     = []   # [{params, results, stats}]
        self.last_narrative= []
        self.checkpoint_path = None

        self._build()

    # ═══ BUILD UI ════════════════════════════════════════════════════

    def _build(self):
        # ── Header ──────────────────────────────────────────────────
        h = tk.Frame(self.root, bg=BG)
        h.pack(fill='x', padx=18, pady=(10,4))
        tk.Label(h, text="⬡  DANTE FILTER v2.0 — FERMI PARADOXON | MONTE CARLO",
                 font=('Courier New',14,'bold'), fg=ACC, bg=BG).pack(side='left')
        tk.Label(h, text="0x44 DANTE | Benjamin Limanovska | Malta 2026",
                 font=('Courier New',8), fg='#333366', bg=BG).pack(side='right')
        tk.Frame(self.root, bg=BRD, height=1).pack(fill='x', padx=18)

        # ── Notebook (Tabs) ─────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('dante.TNotebook', background=BG, borderwidth=0)
        style.configure('dante.TNotebook.Tab', background=BG2, foreground=FG,
                        font=('Courier New',9,'bold'), padding=[14,5])
        style.map('dante.TNotebook.Tab',
                  background=[('selected',ACC)], foreground=[('selected','white')])
        style.configure('dante.Horizontal.TProgressbar',
                        background=ACC, troughcolor='#1a1a2e')
        style.configure('dante.TCombobox',
                        fieldbackground='#1a1a2e', background='#1a1a2e',
                        foreground=FG)

        nb = ttk.Notebook(self.root, style='dante.TNotebook')
        nb.pack(fill='both', expand=True, padx=18, pady=8)

        # Tabs
        t1 = tk.Frame(nb, bg=BG);  nb.add(t1, text='[ SIMULATION ]')
        t2 = tk.Frame(nb, bg=BG);  nb.add(t2, text='[ SENSITIVITÄT ]')
        t3 = tk.Frame(nb, bg=BG);  nb.add(t3, text='[ STATISTIK & EXPORT ]')
        t4 = tk.Frame(nb, bg=BG);  nb.add(t4, text='[ NARRATIV & WAS-WÄRE-WENN ]')

        self._build_tab_sim(t1)
        self._build_tab_sens(t2)
        self._build_tab_stats(t3)
        self._build_tab_narrative(t4)

        # Footer
        tk.Frame(self.root, bg=BRD, height=1).pack(fill='x', padx=18)
        tk.Label(self.root,
                 text='"Sie starben an der Wahrheit." — 0x44 DANTE | Dante Filter 2026',
                 font=('Courier New',6,'italic'), fg='#1a1a3a', bg=BG).pack(pady=4)

    # ═══ TAB 1: SIMULATION ═══════════════════════════════════════════

    def _build_tab_sim(self, parent):
        body = tk.Frame(parent, bg=BG)
        body.pack(fill='both', expand=True)

        # Left panel
        left = tk.Frame(body, bg=BG2, highlightbackground=BRD, highlightthickness=1)
        left.pack(side='left', fill='y', padx=(0,8), ipadx=10, ipady=6)

        tk.Label(left, text="[ PARAMETER ]",
                 font=('Courier New',10,'bold'), fg=ACC, bg=BG2).pack(pady=(8,10))

        sim_opts = ["1.000","10.000","100.000","1.000.000","10.000.000",
                    "100.000.000","1.000.000.000"]
        self.v_sims = self._dropdown(left,"Simulationen:",sim_opts,3)

        self.v_esc    = self._slider(left,"Eskalation (1–10):",1,10,7)
        self.v_actors = self._slider(left,"Akteure:",2,50,10)
        self.v_trust  = self._slider(left,"Vertrauen (%):",0,100,50)
        self.v_info   = self._slider(left,"Information (%):",0,100,80)
        self.v_risk   = self._slider(left,"Risiko (%):",0,100,50)
        self.v_coop   = self._slider(left,"Kooperation (%):",0,100,30)
        self.v_rounds = self._slider(left,"Runden (dyn. Zeit):",1,20,1)
        self.v_shock  = self._slider(left,"Schock-Wahrsch. (%):",0,50,0)

        # Actor type
        tk.Label(left, text="Akteur-Typ:", font=('Courier New',7), fg=FG, bg=BG2
                 ).pack(anchor='w', padx=10, pady=(6,0))
        self.v_atype = tk.StringVar(value="Gemischt")
        cb = ttk.Combobox(left, textvariable=self.v_atype,
                          values=list(ACTOR_TYPES.keys()),
                          state='readonly', width=18, style='dante.TCombobox')
        cb.pack(padx=10, pady=2, fill='x')

        tk.Frame(left, bg=BRD, height=1).pack(fill='x', padx=10, pady=8)

        self.btn_start = tk.Button(left, text="▶  STARTEN",
            font=('Courier New',11,'bold'), bg=ACC, fg='white',
            activebackground='#FF6020', relief='flat', cursor='hand2',
            command=self.start)
        self.btn_start.pack(fill='x', padx=10, pady=3)

        self.btn_stop = tk.Button(left, text="■  STOPP",
            font=('Courier New',9), bg='#222240', fg='#888899',
            relief='flat', cursor='hand2', state='disabled',
            command=self.stop)
        self.btn_stop.pack(fill='x', padx=10, pady=3)

        self.btn_save = tk.Button(left, text="💾  CHECKPOINT",
            font=('Courier New',8), bg='#1a1a3a', fg=FG,
            relief='flat', cursor='hand2', command=self.save_checkpoint)
        self.btn_save.pack(fill='x', padx=10, pady=2)

        self.btn_load = tk.Button(left, text="📂  RESUME",
            font=('Courier New',8), bg='#1a1a3a', fg=FG,
            relief='flat', cursor='hand2', command=self.load_checkpoint)
        self.btn_load.pack(fill='x', padx=10, pady=2)

        tk.Frame(left, bg=BRD, height=1).pack(fill='x', padx=10, pady=6)

        self.pb = ttk.Progressbar(left, style='dante.Horizontal.TProgressbar',
                                  mode='determinate', length=180)
        self.pb.pack(padx=10, pady=3)
        self.lbl_prog  = tk.Label(left, text="0 / 0", font=('Courier New',7),
                                   fg='#555577', bg=BG2)
        self.lbl_prog.pack()
        self.lbl_spd   = tk.Label(left, text="", font=('Courier New',7),
                                   fg='#666688', bg=BG2)
        self.lbl_spd.pack()
        self.lbl_stat  = tk.Label(left, text="Bereit.", font=('Courier New',8),
                                   fg='#555577', bg=BG2, wraplength=180, justify='center')
        self.lbl_stat.pack(pady=6)

        # Right panel
        right = tk.Frame(body, bg=BG)
        right.pack(side='right', fill='both', expand=True)

        # Tiles
        tiles = tk.Frame(right, bg=BG)
        tiles.pack(fill='x', pady=(0,6))
        self.tiles = {}
        for i,(name,col) in enumerate(zip(OUTCOME_NAMES,OUTCOME_COLORS)):
            f = tk.Frame(tiles, bg=BG2, highlightbackground=col, highlightthickness=1)
            f.pack(side='left', fill='both', expand=True, padx=2)
            tk.Label(f, text=name.upper(), font=('Courier New',6,'bold'),
                     fg=col, bg=BG2).pack(pady=(5,0))
            lp = tk.Label(f, text="—", font=('Courier New',18,'bold'), fg=col, bg=BG2)
            lp.pack()
            la = tk.Label(f, text="0", font=('Courier New',6), fg='#555577', bg=BG2)
            la.pack(pady=(0,5))
            self.tiles[i] = (lp, la)

        # Stats row
        srow = tk.Frame(right, bg=BG2, highlightbackground=BRD, highlightthickness=1)
        srow.pack(fill='x', pady=(0,4))
        self.lbl_avg_trust    = self._stat_lbl(srow, "Ø Vertrauen:")
        self.lbl_avg_coop     = self._stat_lbl(srow, "Ø Kooperation:")
        self.lbl_avg_builders = self._stat_lbl(srow, "Ø Eskalierende:")
        self.lbl_std          = self._stat_lbl(srow, "Runs total:")

        # Charts
        self.fig_sim = Figure(figsize=(11,4.2), facecolor=BG)
        self.fig_sim.subplots_adjust(left=0.07,right=0.97,top=0.90,bottom=0.13,wspace=0.38)
        self.ax_pie  = self.fig_sim.add_subplot(131)
        self.ax_line = self.fig_sim.add_subplot(132)
        self.ax_bar  = self.fig_sim.add_subplot(133)
        for ax in [self.ax_pie,self.ax_line,self.ax_bar]:
            ax.set_facecolor(BG2)
        self._empty_sim_charts()

        self.cv_sim = FigureCanvasTkAgg(self.fig_sim, master=right)
        self.cv_sim.get_tk_widget().pack(fill='both', expand=True)

    def _stat_lbl(self, parent, text):
        f = tk.Frame(parent, bg=BG2)
        f.pack(side='left', padx=12, pady=4)
        tk.Label(f, text=text, font=('Courier New',7), fg='#555577', bg=BG2).pack()
        lbl = tk.Label(f, text="—", font=('Courier New',9,'bold'), fg=ACC, bg=BG2)
        lbl.pack()
        return lbl

    # ═══ TAB 2: SENSITIVITÄT ═════════════════════════════════════════

    def _build_tab_sens(self, parent):
        ctrl = tk.Frame(parent, bg=BG2, highlightbackground=BRD, highlightthickness=1)
        ctrl.pack(fill='x', padx=8, pady=6, ipadx=8, ipady=6)

        tk.Label(ctrl, text="[ SENSITIVITÄTS-ANALYSE — HEATMAPS ]",
                 font=('Courier New',10,'bold'), fg=ACC, bg=BG2).pack(pady=(6,8))

        row1 = tk.Frame(ctrl, bg=BG2); row1.pack()
        tk.Label(row1, text="Achse X:", font=('Courier New',8), fg=FG, bg=BG2).pack(side='left',padx=4)
        self.v_sens_x = tk.StringVar(value="trust_pct")
        ttk.Combobox(row1, textvariable=self.v_sens_x,
                     values=["trust_pct","coop_pct","risk_pct","info_pct","escalation"],
                     state='readonly', width=14, style='dante.TCombobox').pack(side='left',padx=4)
        tk.Label(row1, text="Achse Y:", font=('Courier New',8), fg=FG, bg=BG2).pack(side='left',padx=4)
        self.v_sens_y = tk.StringVar(value="coop_pct")
        ttk.Combobox(row1, textvariable=self.v_sens_y,
                     values=["trust_pct","coop_pct","risk_pct","info_pct","escalation"],
                     state='readonly', width=14, style='dante.TCombobox').pack(side='left',padx=4)
        tk.Label(row1, text="Steps:", font=('Courier New',8), fg=FG, bg=BG2).pack(side='left',padx=4)
        self.v_sens_steps = tk.IntVar(value=15)
        tk.Spinbox(row1, textvariable=self.v_sens_steps, from_=5, to=30, width=5,
                   bg='#1a1a2e', fg=FG, insertbackground=FG).pack(side='left',padx=4)
        tk.Label(row1, text="Sims/Zelle:", font=('Courier New',8), fg=FG, bg=BG2).pack(side='left',padx=4)
        self.v_sens_n = tk.IntVar(value=20000)
        tk.Spinbox(row1, textvariable=self.v_sens_n, from_=5000, to=200000, increment=5000,
                   width=8, bg='#1a1a2e', fg=FG, insertbackground=FG).pack(side='left',padx=4)

        btn_row = tk.Frame(ctrl, bg=BG2); btn_row.pack(pady=6)
        tk.Button(btn_row, text="▶  HEATMAP BERECHNEN",
                  font=('Courier New',10,'bold'), bg=ACC, fg='white',
                  relief='flat', cursor='hand2',
                  command=self.run_sensitivity).pack(side='left', padx=6)
        tk.Button(btn_row, text="💾 PNG EXPORT",
                  font=('Courier New',9), bg='#1a1a3a', fg=FG,
                  relief='flat', cursor='hand2',
                  command=lambda: self._export_fig(self.fig_sens, "heatmap")).pack(side='left',padx=6)

        self.lbl_sens_prog = tk.Label(ctrl, text="", font=('Courier New',8),
                                      fg='#666688', bg=BG2)
        self.lbl_sens_prog.pack()

        # Heatmap chart
        self.fig_sens = Figure(figsize=(12,5.5), facecolor=BG)
        self.fig_sens.subplots_adjust(left=0.08,right=0.96,top=0.92,bottom=0.12,wspace=0.35)
        self.ax_heat  = self.fig_sens.add_subplot(121)
        self.ax_heat2 = self.fig_sens.add_subplot(122)
        for ax in [self.ax_heat, self.ax_heat2]:
            ax.set_facecolor(BG2)
        self.cv_sens = FigureCanvasTkAgg(self.fig_sens, master=parent)
        self.cv_sens.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=4)

    # ═══ TAB 3: STATISTIK & EXPORT ═══════════════════════════════════

    def _build_tab_stats(self, parent):
        top = tk.Frame(parent, bg=BG)
        top.pack(fill='x', padx=8, pady=6)

        tk.Label(top, text="[ STATISTIK & EXPORT ]",
                 font=('Courier New',10,'bold'), fg=ACC, bg=BG).pack(anchor='w')

        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill='x', padx=8, pady=4)
        for txt, cmd in [("📊 CSV EXPORT", self.export_csv),
                          ("📋 JSON EXPORT", self.export_json),
                          ("🖼  CHARTS PNG",  lambda: self._export_fig(self.fig_sim,"sim_charts")),
                          ("📈 BOXPLOT",       self.show_boxplot)]:
            tk.Button(btn_row, text=txt, font=('Courier New',9,'bold'),
                      bg='#1a1a3a', fg=FG, activebackground=ACC,
                      relief='flat', cursor='hand2', command=cmd,
                      padx=10, pady=5).pack(side='left', padx=6)

        # Stats text
        self.txt_stats = tk.Text(parent, bg=BG2, fg=FG,
                                  font=('Courier New',9), relief='flat',
                                  wrap='word', height=12)
        self.txt_stats.pack(fill='x', padx=8, pady=4)

        # Boxplot area
        self.fig_box = Figure(figsize=(12,4), facecolor=BG)
        self.fig_box.subplots_adjust(left=0.08,right=0.97,top=0.90,bottom=0.15,wspace=0.3)
        self.ax_box = self.fig_box.add_subplot(111)
        self.ax_box.set_facecolor(BG2)
        self.cv_box = FigureCanvasTkAgg(self.fig_box, master=parent)
        self.cv_box.get_tk_widget().pack(fill='both', expand=True, padx=8)

    # ═══ TAB 4: NARRATIV & WAS-WÄRE-WENN ════════════════════════════

    def _build_tab_narrative(self, parent):
        tk.Label(parent, text="[ NARRATIV & WAS-WÄRE-WENN-MODUS ]",
                 font=('Courier New',10,'bold'), fg=ACC, bg=BG).pack(anchor='w', padx=10, pady=8)

        # What-if controls
        wif = tk.LabelFrame(parent, text=" WAS-WÄRE-WENN ", bg=BG2, fg=ACC,
                             font=('Courier New',9,'bold'))
        wif.pack(fill='x', padx=8, pady=4)

        row = tk.Frame(wif, bg=BG2); row.pack(pady=6)
        self.wif_sliders = {}
        for txt, key, val in [("Vertrauen %", "trust_pct", 80),
                               ("Kooperation %","coop_pct", 70),
                               ("Eskalation",   "escalation",4),
                               ("Schock %",     "shock_prob",5)]:
            f = tk.Frame(row, bg=BG2); f.pack(side='left', padx=12)
            tk.Label(f, text=txt, font=('Courier New',7), fg=FG, bg=BG2).pack()
            vl = tk.Label(f, text=str(val), font=('Courier New',8,'bold'), fg=ACC, bg=BG2)
            vl.pack()
            mx = 100 if "%" in txt else (10 if txt=="Eskalation" else 50)
            var = tk.IntVar(value=val)
            def upd(v, lbl=vl): lbl.config(text=str(int(float(v))))
            tk.Scale(f, variable=var, from_=0, to=mx, orient='horizontal',
                     showvalue=False, bg=BG2, fg=ACC, troughcolor='#1a1a3a',
                     highlightthickness=0, relief='flat', length=100,
                     command=upd).pack()
            self.wif_sliders[key] = var

        tk.Button(wif, text="▶  WAS-WÄRE-WENN SIMULIEREN (1M)",
                  font=('Courier New',10,'bold'), bg=ACC, fg='white',
                  relief='flat', cursor='hand2',
                  command=self.run_whatif).pack(pady=6)

        self.lbl_wif = tk.Label(wif, text="", font=('Courier New',9,'bold'),
                                 fg='#00FF7F', bg=BG2)
        self.lbl_wif.pack(pady=4)

        # Narrative output
        nar_f = tk.LabelFrame(parent, text=" NARRATIV — BEISPIELSIMULATION ",
                               bg=BG2, fg=ACC, font=('Courier New',9,'bold'))
        nar_f.pack(fill='both', expand=True, padx=8, pady=6)

        tk.Button(nar_f, text="▶  NARRATIVE GENERIEREN",
                  font=('Courier New',9,'bold'), bg='#1a1a3a', fg=FG,
                  relief='flat', cursor='hand2',
                  command=self.generate_narrative).pack(pady=6)

        self.txt_nar = tk.Text(nar_f, bg=BG, fg=FG, font=('Courier New',8),
                                relief='flat', wrap='word')
        sb = ttk.Scrollbar(nar_f, command=self.txt_nar.yview)
        self.txt_nar.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.txt_nar.pack(fill='both', expand=True, padx=4, pady=4)

    # ═══ SIMULATION LOGIC ════════════════════════════════════════════

    def _get_params(self):
        return dict(
            escalation = self.v_esc.get(),
            n_actors   = self.v_actors.get(),
            trust_pct  = float(self.v_trust.get()),
            info_pct   = float(self.v_info.get()),
            risk_pct   = float(self.v_risk.get()),
            coop_pct   = float(self.v_coop.get()),
            actor_type = self.v_atype.get(),
            n_rounds   = self.v_rounds.get(),
            shock_prob = self.v_shock.get() / 100.0,
            shock_strength = 0.3,
        )

    def start(self):
        s = self.v_sims.get().replace('.','').replace(',','')
        self.target = int(s)
        self.done   = 0
        self.results= np.zeros(4, dtype=np.int64)
        self.hist_collapse = []
        self.hist_x        = []
        self.running = True
        self.t0      = time.time()
        self.pb['value']=0
        self.pb['maximum']=100
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self._empty_sim_charts()
        params = self._get_params()
        threading.Thread(target=self._worker, args=(params,), daemon=True).start()
        self._poll()

    def _worker(self, params):
        rem = self.target
        bn  = 0
        cum_stats = dict(avg_trust=[], avg_coop=[], avg_builders=[])
        while rem > 0 and self.running:
            b = min(BATCH, rem)
            counts, stats = simulate_batch(b, **params)
            self.results += counts
            self.done    += b
            rem          -= b
            bn           += 1
            cr = 100.0 * self.results[0] / max(self.done,1)
            self.hist_x.append(bn)
            self.hist_collapse.append(cr)
            for k in cum_stats:
                cum_stats[k].append(stats[k])
        # Save run to history
        self.all_stats.append({
            'params':  params,
            'results': self.results.copy().tolist(),
            'total':   self.done,
            'cum_stats': cum_stats,
        })
        self.running = False

    def _poll(self):
        self._refresh_tiles()
        self._refresh_status()
        if self.done > 0 and (len(self.hist_x) % 3 == 0 or not self.running):
            self._update_sim_charts()
        if self.running:
            self.root.after(250, self._poll)
        else:
            self._refresh_tiles()
            self._update_sim_charts()
            self._on_done()

    def _refresh_tiles(self):
        total = max(self.done,1)
        for i in range(4):
            pct = 100.0 * self.results[i] / total
            lp, la = self.tiles[i]
            lp.config(text=f"{pct:.2f}%")
            la.config(text=self._fn(self.results[i]))

    def _refresh_status(self):
        if self.target > 0 and self.done > 0:
            pct = 100.0 * self.done / self.target
            self.pb['value'] = pct
            self.lbl_prog.config(text=f"{self._fn(self.done)} / {self._fn(self.target)}")
            if self.t0:
                el  = max(time.time()-self.t0, 0.001)
                spd = self.done/el
                ss  = f"{spd/1e6:.1f}M/s" if spd>=1e6 else f"{spd/1e3:.0f}K/s"
                eta = (self.target-self.done)/max(spd,1)
                self.lbl_spd.config(text=f"⚡ {ss}  ETA {eta:.0f}s")
            cr = 100.0*self.results[0]/max(self.done,1)
            self.lbl_stat.config(
                text=f"{'▶ LÄUFT' if self.running else '✓ FERTIG'}\nKollaps: {cr:.2f}%",
                fg=ACC if cr>90 else '#00FF7F')

    def _on_done(self):
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        # Update stats labels
        if self.all_stats:
            last = self.all_stats[-1]
            cs   = last['cum_stats']
            self.lbl_avg_trust.config(text=f"{np.mean(cs['avg_trust']):.2f}")
            self.lbl_avg_coop.config(text=f"{np.mean(cs['avg_coop']):.2f}")
            self.lbl_avg_builders.config(text=f"{np.mean(cs['avg_builders']):.2f}")
            self.lbl_std.config(text=self._fn(self.done))
        self._update_stats_text()

    def stop(self):
        self.running = False
        self.btn_stop.config(state='disabled')

    # ═══ CHARTS ══════════════════════════════════════════════════════

    def _empty_sim_charts(self):
        for ax in [self.ax_pie,self.ax_line,self.ax_bar]:
            ax.clear(); ax.set_facecolor(BG2)
            for sp in ax.spines.values(): sp.set_color(BRD)
        self.ax_pie.text(0.5,0.5,'Noch keine Daten',ha='center',va='center',
                          color='#333366',fontsize=9,transform=self.ax_pie.transAxes,
                          fontfamily='monospace')
        self.ax_pie.axis('off')
        self.ax_pie.set_title('Outcome-Verteilung',color=FG,fontsize=8,fontfamily='monospace')
        self.ax_line.axhline(y=98.7,color='#FF2020',ls='--',lw=0.8,alpha=0.5,label='98.7%')
        self.ax_line.set_ylim(0,100)
        self.ax_line.set_title('Kollapsrate (live)',color=FG,fontsize=8,fontfamily='monospace')
        self.ax_line.tick_params(colors='#555577',labelsize=6)
        self.ax_bar.set_title('Verteilung (%)',color=FG,fontsize=8,fontfamily='monospace')
        self.ax_bar.tick_params(colors='#555577',labelsize=6)
        try: self.cv_sim.draw()
        except: pass

    def _update_sim_charts(self):
        total = max(self.done,1)
        vals  = self.results.copy()

        # Pie
        self.ax_pie.clear(); self.ax_pie.set_facecolor(BG2)
        if vals.sum() > 0:
            mask = vals>0
            self.ax_pie.pie(vals[mask],
                colors=[OUTCOME_COLORS[i] for i in range(4) if mask[i]],
                labels=[f"{OUTCOME_NAMES[i]}\n{100*vals[i]/total:.2f}%"
                        for i in range(4) if mask[i]],
                startangle=90,
                wedgeprops={'linewidth':0.5,'edgecolor':BG},
                textprops={'color':FG,'fontsize':6.5,'fontfamily':'monospace'})
        self.ax_pie.set_title('Outcome-Verteilung',color=FG,fontsize=8,fontfamily='monospace')

        # Line
        self.ax_line.clear(); self.ax_line.set_facecolor(BG2)
        for sp in self.ax_line.spines.values(): sp.set_color(BRD)
        if self.hist_x:
            self.ax_line.plot(self.hist_x,self.hist_collapse,color='#FF2020',lw=1.2)
            self.ax_line.fill_between(self.hist_x,self.hist_collapse,alpha=0.12,color='#FF2020')
            cur = self.hist_collapse[-1]
            self.ax_line.axhline(y=98.7,color='#FF4500',ls='--',lw=0.8,alpha=0.5,label='98.7%')
            self.ax_line.axhline(y=cur, color='#FFD700',ls=':',lw=0.8,alpha=0.7,label=f'{cur:.1f}%')
            self.ax_line.legend(fontsize=6,facecolor=BG2,labelcolor=FG,edgecolor=BRD)
        self.ax_line.set_ylim(0,100)
        self.ax_line.set_title('Kollapsrate (live)',color=FG,fontsize=8,fontfamily='monospace')
        self.ax_line.tick_params(colors='#555577',labelsize=6)
        self.ax_line.grid(True,alpha=0.08,color=BRD)

        # Bar
        self.ax_bar.clear(); self.ax_bar.set_facecolor(BG2)
        for sp in self.ax_bar.spines.values(): sp.set_color(BRD)
        if vals.sum()>0:
            pcts = [100.0*v/total for v in vals]
            bars = self.ax_bar.bar(range(4),pcts,color=OUTCOME_COLORS,
                                    width=0.6,edgecolor=BG,linewidth=0.5)
            for bar,p in zip(bars,pcts):
                if p>0:
                    self.ax_bar.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.8,
                                     f"{p:.1f}%",ha='center',va='bottom',
                                     color=FG,fontsize=6,fontfamily='monospace')
        self.ax_bar.set_xticks(range(4))
        self.ax_bar.set_xticklabels(['Kollaps','Dystopie','Verzicht','Sprung'],
                                     fontsize=6,color=FG,fontfamily='monospace')
        self.ax_bar.set_ylim(0,105)
        self.ax_bar.tick_params(colors='#555577',labelsize=6)
        self.ax_bar.set_title('Verteilung (%)',color=FG,fontsize=8,fontfamily='monospace')
        self.ax_bar.grid(True,alpha=0.08,color=BRD,axis='y')

        try: self.cv_sim.draw()
        except: pass

    # ═══ SENSITIVITÄT ════════════════════════════════════════════════

    def run_sensitivity(self):
        px = self.v_sens_x.get()
        py = self.v_sens_y.get()
        if px == py:
            messagebox.showwarning("Fehler","X und Y müssen verschieden sein!"); return

        steps = self.v_sens_steps.get()
        n     = self.v_sens_n.get()
        params_base = self._get_params()

        rng_map = {
            'trust_pct':  np.linspace(0,100,steps),
            'coop_pct':   np.linspace(0,100,steps),
            'risk_pct':   np.linspace(0,100,steps),
            'info_pct':   np.linspace(0,100,steps),
            'escalation': np.linspace(1,10,steps,dtype=int),
        }
        ax_vals = rng_map[px]
        ay_vals = rng_map[py]
        self.lbl_sens_prog.config(text=f"Berechne {steps}×{steps}={steps**2} Zellen...")
        self.root.update()

        def run():
            fixed = dict(params_base)
            def cb(done, total, hm):
                pct = int(100*done/total)
                self.lbl_sens_prog.config(text=f"Fortschritt: {done}/{total} ({pct}%)")
            hm = sensitivity_scan(ax_vals, ay_vals, fixed, n, px, py, callback=cb)
            self.root.after(0, lambda: self._draw_heatmap(hm, ax_vals, ay_vals, px, py))

        threading.Thread(target=run, daemon=True).start()

    def _draw_heatmap(self, hm, ax_vals, ay_vals, px, py):
        self.ax_heat.clear(); self.ax_heat2.clear()
        for ax in [self.ax_heat, self.ax_heat2]:
            ax.set_facecolor(BG2)
            for sp in ax.spines.values(): sp.set_color(BRD)

        im = self.ax_heat.imshow(hm, origin='lower', aspect='auto',
                                  cmap='RdYlGn_r', vmin=0, vmax=100)
        self.fig_sens.colorbar(im, ax=self.ax_heat, label='Kollaps %')
        steps = len(ax_vals)
        ticks = list(range(0, steps, max(1, steps//5)))
        self.ax_heat.set_xticks(ticks)
        self.ax_heat.set_xticklabels([f"{ay_vals[i]:.0f}" for i in ticks],
                                      fontsize=6, color=FG, fontfamily='monospace')
        self.ax_heat.set_yticks(ticks)
        self.ax_heat.set_yticklabels([f"{ax_vals[i]:.0f}" for i in ticks],
                                      fontsize=6, color=FG, fontfamily='monospace')
        self.ax_heat.set_xlabel(py, color=FG, fontsize=7, fontfamily='monospace')
        self.ax_heat.set_ylabel(px, color=FG, fontsize=7, fontfamily='monospace')
        self.ax_heat.set_title(f'Kollapsrate: {px} vs {py}',
                                color=FG, fontsize=8, fontfamily='monospace')

        # Linienchart: durchschnittliche Kollapsrate pro X-Wert
        avg_per_row = hm.mean(axis=1)
        self.ax_heat2.plot(ax_vals, avg_per_row, color=OUTCOME_COLORS[0], lw=2)
        self.ax_heat2.fill_between(ax_vals, avg_per_row, alpha=0.2, color=OUTCOME_COLORS[0])
        self.ax_heat2.axhline(y=98.7, color='#FFD700', ls='--', lw=1, label='98.7%')
        self.ax_heat2.set_ylim(0,100)
        self.ax_heat2.set_xlabel(px, color=FG, fontsize=7, fontfamily='monospace')
        self.ax_heat2.set_ylabel('Ø Kollaps %', color=FG, fontsize=7, fontfamily='monospace')
        self.ax_heat2.tick_params(colors='#555577', labelsize=6)
        self.ax_heat2.set_title(f'Kollaps vs {px}',
                                 color=FG, fontsize=8, fontfamily='monospace')
        self.ax_heat2.grid(True, alpha=0.1, color=BRD)
        self.ax_heat2.legend(fontsize=6, facecolor=BG2, labelcolor=FG, edgecolor=BRD)

        self.lbl_sens_prog.config(
            text=f"✓ Fertig | Min: {hm.min():.1f}%  Max: {hm.max():.1f}%  Ø: {hm.mean():.1f}%")
        try: self.cv_sens.draw()
        except: pass

    # ═══ STATISTIK & EXPORT ══════════════════════════════════════════

    def _update_stats_text(self):
        if not self.all_stats: return
        last  = self.all_stats[-1]
        total = max(last['total'],1)
        res   = last['results']
        p     = last['params']
        cs    = last['cum_stats']
        lines = [
            "╔══════════════════════════════════════════════════",
            "║  DANTE <DILEMMA> — STATISTISCHE ZUSAMMENFASSUNG",
            "╠══════════════════════════════════════════════════",
            f"║  Simulationen    : {self._fn(last['total'])}",
            f"║  Akteur-Typ      : {p['actor_type']}",
            f"║  Runden          : {p['n_rounds']}",
            f"║  Eskalation      : {p['escalation']}",
            f"║  Vertrauen       : {p['trust_pct']}%",
            f"║  Kooperation     : {p['coop_pct']}%",
            "╠══════════════════════════════════════════════════",
        ]
        for i,(n,c) in enumerate(zip(OUTCOME_NAMES,OUTCOME_COLORS)):
            pct = 100.0*res[i]/total
            lines.append(f"║  {n:<22}: {pct:>6.2f}%  ({self._fn(res[i])})")
        collapse_rate = 100.0*(res[0]+res[1])/total
        lines += [
            "╠══════════════════════════════════════════════════",
            f"║  Kollaps+Dystopie: {collapse_rate:.2f}%",
            f"║  Ø Vertrauen     : {np.mean(cs['avg_trust']):.3f}",
            f"║  Ø Kooperation   : {np.mean(cs['avg_coop']):.3f}",
            f"║  Ø Eskalierende  : {np.mean(cs['avg_builders']):.2f}",
            "╚══════════════════════════════════════════════════",
        ]
        self.txt_stats.delete('1.0','end')
        self.txt_stats.insert('end', '\n'.join(lines))

    def show_boxplot(self):
        if not self.all_stats:
            messagebox.showinfo("Info","Erst eine Simulation starten!"); return
        self.ax_box.clear(); self.ax_box.set_facecolor(BG2)
        for sp in self.ax_box.spines.values(): sp.set_color(BRD)

        data = []
        for run in self.all_stats:
            total = max(run['total'],1)
            data.append([100.0*run['results'][i]/total for i in range(4)])
        data = list(zip(*data))  # transpose

        bp = self.ax_box.boxplot(
            [list(d) for d in data],
            patch_artist=True,
            medianprops=dict(color='white',lw=2),
            whiskerprops=dict(color=FG),
            capprops=dict(color=FG),
            flierprops=dict(marker='o',color=FG,markersize=3))
        for patch,col in zip(bp['boxes'], OUTCOME_COLORS):
            patch.set_facecolor(col); patch.set_alpha(0.7)

        self.ax_box.set_xticks(range(1,5))
        self.ax_box.set_xticklabels(OUTCOME_NAMES, fontsize=7,
                                     color=FG, fontfamily='monospace')
        self.ax_box.tick_params(colors='#555577', labelsize=7)
        self.ax_box.set_ylabel('Häufigkeit %', color=FG, fontsize=8)
        self.ax_box.set_title('Verteilung über alle Runs',
                               color=FG, fontsize=9, fontfamily='monospace')
        self.ax_box.grid(True, alpha=0.08, color=BRD, axis='y')
        try: self.cv_box.draw()
        except: pass

    def export_csv(self):
        if not self.all_stats:
            messagebox.showinfo("Info","Keine Daten!"); return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
               filetypes=[("CSV","*.csv")], title="CSV speichern")
        if not path: return
        with open(path,'w',newline='',encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['run','total','kollaps_pct','dystopie_pct','verzicht_pct','sprung_pct',
                        'escalation','n_actors','trust_pct','coop_pct','risk_pct',
                        'actor_type','n_rounds'])
            for i,run in enumerate(self.all_stats):
                total = max(run['total'],1)
                res   = run['results']
                p     = run['params']
                w.writerow([i+1, run['total'],
                             100*res[0]/total, 100*res[1]/total,
                             100*res[2]/total, 100*res[3]/total,
                             p['escalation'], p['n_actors'], p['trust_pct'],
                             p['coop_pct'], p['risk_pct'], p['actor_type'], p['n_rounds']])
        messagebox.showinfo("✓","CSV gespeichert: "+path)

    def export_json(self):
        if not self.all_stats:
            messagebox.showinfo("Info","Keine Daten!"); return
        path = filedialog.asksaveasfilename(defaultextension=".json",
               filetypes=[("JSON","*.json")], title="JSON speichern")
        if not path: return
        out = []
        for run in self.all_stats:
            total = max(run['total'],1)
            out.append({
                'total': run['total'],
                'outcomes': {OUTCOME_NAMES[i]: {'count':run['results'][i],
                             'pct':100*run['results'][i]/total} for i in range(4)},
                'params':  run['params'],
            })
        with open(path,'w',encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        messagebox.showinfo("✓","JSON gespeichert: "+path)

    def _export_fig(self, fig, name):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg")],
            initialfile=f"dante_{name}.png",
            title="Chart speichern")
        if path:
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
            messagebox.showinfo("✓","Gespeichert: "+path)

    # ═══ CHECKPOINT ══════════════════════════════════════════════════

    def save_checkpoint(self):
        path = filedialog.asksaveasfilename(defaultextension=".dante",
               filetypes=[("Dante Checkpoint","*.dante")], title="Checkpoint speichern")
        if not path: return
        state = dict(results=self.results, done=self.done,
                     target=self.target, hist_x=self.hist_x,
                     hist_collapse=self.hist_collapse,
                     params=self._get_params(), all_stats=self.all_stats)
        with open(path,'wb') as f: pickle.dump(state,f)
        messagebox.showinfo("✓","Checkpoint gespeichert: "+path)

    def load_checkpoint(self):
        path = filedialog.askopenfilename(
               filetypes=[("Dante Checkpoint","*.dante")], title="Checkpoint laden")
        if not path: return
        with open(path,'rb') as f: state = pickle.load(f)
        self.results        = state['results']
        self.done           = state['done']
        self.target         = state['target']
        self.hist_x         = state['hist_x']
        self.hist_collapse  = state['hist_collapse']
        self.all_stats      = state.get('all_stats',[])
        self._refresh_tiles()
        self._update_sim_charts()
        self._update_stats_text()
        messagebox.showinfo("✓",f"Geladen: {self._fn(self.done)} Simulationen")

    # ═══ WAS-WÄRE-WENN ═══════════════════════════════════════════════

    def run_whatif(self):
        base_p = self._get_params()
        base_p['trust_pct']  = float(self.wif_sliders['trust_pct'].get())
        base_p['coop_pct']   = float(self.wif_sliders['coop_pct'].get())
        base_p['escalation'] = self.wif_sliders['escalation'].get()
        base_p['shock_prob'] = self.wif_sliders['shock_prob'].get() / 100.0

        self.lbl_wif.config(text="Berechne...", fg='#FFD700')
        self.root.update()

        def run():
            counts,stats = simulate_batch(1_000_000, **base_p)
            total = max(counts.sum(),1)
            cr    = 100.0*counts[0]/total
            dr    = 100.0*counts[1]/total
            fr    = 100.0*(counts[2]+counts[3])/total
            msg   = (f"✓  Kollaps: {cr:.1f}%  |  Dystopie: {dr:.1f}%  |  "
                     f"Überleben: {fr:.2f}%  |  1.000.000 Simulationen")
            col   = '#00FF7F' if cr < 80 else (ACC if cr < 95 else '#FF2020')
            self.root.after(0, lambda: self.lbl_wif.config(text=msg, fg=col))

        threading.Thread(target=run, daemon=True).start()

    # ═══ NARRATIV ════════════════════════════════════════════════════

    def generate_narrative(self):
        """Generiert eine schrittweise Narrative-Simulation"""
        params = self._get_params()
        esc    = params['escalation']
        n_a    = params['n_actors']
        trust  = params['trust_pct']
        coop   = params['coop_pct']
        rounds = max(params['n_rounds'], 6)

        self.txt_nar.delete('1.0','end')

        def phase_name(r, total):
            pct = r/total
            if pct < 0.17: return "Erkenntnis"
            if pct < 0.33: return "Paranoia"
            if pct < 0.50: return "Rationalitätsfalle"
            if pct < 0.67: return "Eskalation"
            if pct < 0.83: return "Punkt ohne Wiederkehr"
            return "Katastrophe"

        actor_names = [f"Akteur-{i+1}" for i in range(min(n_a,8))]
        rng = np.random.default_rng()

        lines = [
            "╔══════════════════════════════════════════════════════════════",
            "║  DANTE FILTER — NARRATIVE SIMULATION",
            f"║  Eskalation: {esc}/10 | Akteure: {n_a} | Vertrauen: {trust}% | Runden: {rounds}",
            "╚══════════════════════════════════════════════════════════════\n",
        ]

        cur_trust = trust/100
        cur_coop  = coop /100
        escalated_actors = set()

        for rnd in range(1, rounds+1):
            phase = phase_name(rnd, rounds)
            paranoia = (esc/10.0)**1.7 * (1 + (rnd/rounds)*0.5)

            # Entscheidungen
            decisions = {}
            for a in actor_names:
                t = np.clip(rng.normal(cur_trust, 0.12), 0, 1)
                c = np.clip(rng.normal(cur_coop,  0.10), 0, 1)
                r = np.clip(rng.normal(0.5, 0.10), 0, 1)
                p = 0.35*(1-t)*paranoia + 0.45*(1-(t*c)**2.2)*(esc/10) + 0.20*(1-c)*r
                decisions[a] = rng.random() < np.clip(p,0,1)
                if decisions[a]: escalated_actors.add(a)

            n_esc = sum(decisions.values())
            lines.append(f"─── RUNDE {rnd:2d} | PHASE: {phase.upper()} ───────────────────────")

            for a,d in decisions.items():
                action = "🔴 ESKALIERT" if d else "🟢 kooperiert"
                lines.append(f"  {a:<12}: {action}")

            if n_esc > 0:
                triggers = [a for a,d in decisions.items() if d]
                lines.append(f"\n  ⚠  {', '.join(triggers[:3])} eskalieren → "
                              f"Misstrauen steigt um {n_esc*8:.0f}%")

            # Update trust/coop
            cur_trust = max(0, cur_trust - n_esc*0.08)
            cur_coop  = max(0, cur_coop  - n_esc*0.06)
            lines.append(f"  Ø Vertrauen: {cur_trust*100:.1f}%  |  "
                          f"Ø Kooperation: {cur_coop*100:.1f}%\n")

        # Outcome
        frac = len(escalated_actors)/len(actor_names)
        if frac > 0.5:
            outcome = "💀 TOTALER KOLLAPS — Die Zivilisation zerstört sich selbst."
            fermi   = ("→ FERMI-ERKLÄRUNG: Diese Zivilisation sendet keine Signale mehr. "
                       "Das Universum empfängt Stille.")
        elif frac > 0:
            outcome = "⚠  DIGITALE DYSTOPIE — Ein Akteur kontrolliert alle anderen."
            fermi   = ("→ FERMI-ERKLÄRUNG: Signale hören auf — intern unterdrückt. "
                       "Von außen: Stille.")
        else:
            outcome = "✅ FREIWILLIGER VERZICHT — Instabiler Frieden möglich."
            fermi   = "→ FERMI-ERKLÄRUNG: Diese Zivilisation könnte Signale senden."

        lines += [
            "═══════════════════════════════════════════════════════════════",
            f"  OUTCOME: {outcome}",
            f"  {fermi}",
            "═══════════════════════════════════════════════════════════════",
        ]

        self.txt_nar.insert('end', '\n'.join(lines))
        self.txt_nar.see('end')

    # ═══ HELPERS ═════════════════════════════════════════════════════

    def _dropdown(self, parent, label, options, idx=0):
        tk.Label(parent, text=label, font=('Courier New',7), fg=FG, bg=BG2
                 ).pack(anchor='w', padx=10, pady=(5,0))
        var = tk.StringVar(value=options[idx])
        ttk.Combobox(parent, textvariable=var, values=options,
                     state='readonly', width=20,
                     style='dante.TCombobox').pack(padx=10, pady=1, fill='x')
        return var

    def _slider(self, parent, label, lo, hi, default):
        frame = tk.Frame(parent, bg=BG2); frame.pack(fill='x', padx=10, pady=1)
        row   = tk.Frame(frame, bg=BG2); row.pack(fill='x')
        tk.Label(row, text=label, font=('Courier New',7), fg=FG, bg=BG2).pack(side='left')
        vl = tk.Label(row, text=str(default), font=('Courier New',7,'bold'), fg=ACC, bg=BG2)
        vl.pack(side='right')
        var = tk.IntVar(value=default)
        def upd(v, l=vl): l.config(text=str(int(float(v))))
        tk.Scale(frame, variable=var, from_=lo, to=hi, orient='horizontal',
                 showvalue=False, bg=BG2, fg=ACC, troughcolor='#1a1a3a',
                 highlightthickness=0, relief='flat', command=upd).pack(fill='x')
        return var

    @staticmethod
    def _fn(n):
        if n>=1e9: return f"{n/1e9:.2f}B"
        if n>=1e6: return f"{n/1e6:.1f}M"
        if n>=1e3: return f"{n/1e3:.1f}K"
        return str(int(n))


# ═══ MAIN ════════════════════════════════════════════════════════════

def main():
    for pkg in ('numpy','matplotlib'):
        try: __import__(pkg)
        except ImportError:
            print(f"pip install {pkg}"); sys.exit(1)
    root = tk.Tk()
    DanteV2(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

if __name__ == "__main__":
    main()