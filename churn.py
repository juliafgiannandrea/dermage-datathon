"""
ANÁLISE DE CHURN — Dermage Datathon Girls in Tech 2026
Threshold: 180 dias sem compra = churn
Inclui: taxa, perfil, RFM, sobrevivência, predição
Input: dermage_merge_sku.csv + dermage_rfm_clientes.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
ARQUIVO_MERGE = "dermage_merge_sku.csv"
ARQUIVO_RFM   = "dermage_rfm_clientes.csv"
CHURN_DIAS    = 220
DATA_REF_STR  = None   # None = usa a data máxima da base automaticamente

COR_CHURN  = "#E75480"
COR_ATIVO  = "#2E8B57"
COR_BASE   = "#2E4057"
COR_ROXO   = "#6A5ACD"
PALETTE    = [COR_CHURN, COR_ATIVO, COR_ROXO, "#F4A261", "#264653", "#E9C46A"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})

# ──────────────────────────────────────────
# 1. CARREGAMENTO
# ──────────────────────────────────────────
print("=" * 60)
print("1. CARREGANDO DADOS")
print("=" * 60)

df = pd.read_csv(ARQUIVO_MERGE, low_memory=False)
df["data_pedido"] = pd.to_datetime(df["data_pedido"], errors="coerce")

# Consistência de cli_document
if "cli_document_x" in df.columns:
    df["cli_document"] = df["cli_document_x"].fillna(df.get("cli_document_y", ""))

# Data de referência (hoje relativo à base)
DATA_REF = pd.to_datetime(DATA_REF_STR) if DATA_REF_STR else df["data_pedido"].max()
print(f"Data de referência: {DATA_REF.date()}")
print(f"Threshold de churn: {CHURN_DIAS} dias")

# 1 linha por pedido por cliente
df_ped = (
    df.dropna(subset=["cli_document", "data_pedido"])
      .drop_duplicates(subset=["cli_document", "order_id"])
      [["cli_document", "order_id", "data_pedido", "valor_pedido",
        "tipo_cliente", "canal", "uf"]]
      .copy()
)

print(f"\nPedidos únicos: {len(df_ped):,}")
print(f"Clientes únicos: {df_ped['cli_document'].nunique():,}")

# ──────────────────────────────────────────
# 2. LABEL DE CHURN POR CLIENTE
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DEFININDO CHURN (threshold = 180 dias)")
print("=" * 60)

clientes = df_ped.groupby("cli_document").agg(
    primeira_compra = ("data_pedido", "min"),
    ultima_compra   = ("data_pedido", "max"),
    n_pedidos       = ("order_id",    "count"),
    ticket_total    = ("valor_pedido","sum"),
    ticket_medio    = ("valor_pedido","mean"),
    tipo_cliente    = ("tipo_cliente","first"),
    canal           = ("canal",       "first"),
    uf              = ("uf",          "first"),
).reset_index()

# Dias desde a última compra
clientes["dias_desde_ultima"] = (DATA_REF - clientes["ultima_compra"]).dt.days

# Churn = não comprou nos últimos CHURN_DIAS dias
# Mas só classifica clientes que tiveram tempo suficiente para churnar
# (entraram há mais de CHURN_DIAS dias)
clientes["dias_desde_primeira"] = (DATA_REF - clientes["primeira_compra"]).dt.days
clientes_elegíveis = clientes[clientes["dias_desde_primeira"] >= CHURN_DIAS].copy()

clientes_elegíveis["churn"] = (
    clientes_elegíveis["dias_desde_ultima"] >= CHURN_DIAS
).astype(int)

# Tempo de vida (em dias)
clientes_elegíveis["tempo_vida"] = clientes_elegíveis["dias_desde_primeira"]

taxa_churn = clientes_elegíveis["churn"].mean() * 100
n_churn    = clientes_elegíveis["churn"].sum()
n_ativo    = len(clientes_elegíveis) - n_churn

print(f"\nClientes elegíveis (entrada há >{CHURN_DIAS} dias): {len(clientes_elegíveis):,}")
print(f"Churned:  {n_churn:,} ({taxa_churn:.1f}%)")
print(f"Ativos:   {n_ativo:,} ({100-taxa_churn:.1f}%)")

# ──────────────────────────────────────────
# 3. PERFIL DO CLIENTE CHURNED
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PERFIL DO CLIENTE CHURNED vs ATIVO")
print("=" * 60)

perfil = clientes_elegíveis.groupby("churn").agg(
    clientes      = ("cli_document",    "count"),
    ticket_medio  = ("ticket_medio",    "mean"),
    ticket_total  = ("ticket_total",    "mean"),
    n_pedidos_med = ("n_pedidos",       "mean"),
    dias_vida_med = ("tempo_vida",      "mean"),
).round(2)
perfil.index = ["Ativo", "Churned"]
print(f"\n{perfil}")

print(f"\nChurn por tipo de cliente:")
print(clientes_elegíveis.groupby("tipo_cliente")["churn"].mean().apply(
    lambda x: f"{x*100:.1f}%"))

print(f"\nChurn por canal:")
print(clientes_elegíveis.groupby("canal")["churn"].mean().apply(
    lambda x: f"{x*100:.1f}%"))

print(f"\nChurn por UF (top 8):")
uf_churn = clientes_elegíveis.groupby("uf")["churn"].agg(["mean","count"])
uf_churn = uf_churn[uf_churn["count"] >= 100].sort_values("mean", ascending=False)
print(uf_churn["mean"].apply(lambda x: f"{x*100:.1f}%").head(8))

# ──────────────────────────────────────────
# 4. CHURN POR SEGMENTO RFM
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CHURN POR SEGMENTO RFM")
print("=" * 60)

try:
    rfm = pd.read_csv(ARQUIVO_RFM)
    churn_rfm = clientes_elegíveis.merge(
        rfm[["cli_document", "segmento", "R", "F", "M", "rfm_total"]],
        on="cli_document", how="left"
    )
    seg_churn = churn_rfm.groupby("segmento").agg(
        clientes     = ("cli_document", "count"),
        taxa_churn   = ("churn",        "mean"),
        ticket_medio = ("ticket_medio", "mean"),
    ).round(3)
    seg_churn["taxa_churn_%"] = (seg_churn["taxa_churn"] * 100).round(1)
    seg_churn = seg_churn.sort_values("taxa_churn", ascending=False)
    print(f"\n{seg_churn[['clientes','taxa_churn_%','ticket_medio']]}")
    TEM_RFM = True
except FileNotFoundError:
    print("⚠️  dermage_rfm_clientes.csv não encontrado — pulando seção RFM.")
    churn_rfm = clientes_elegíveis.copy()
    TEM_RFM = False

# ──────────────────────────────────────────
# 5. CURVA DE SOBREVIVÊNCIA (Kaplan-Meier simplificado)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CURVA DE SOBREVIVÊNCIA")
print("=" * 60)

# Para cada cliente: tempo até o churn ou última observação (censored)
# Evento = churn (1); censored = ainda ativo (0)
df_surv = clientes_elegíveis[["cli_document","dias_desde_ultima","churn","tipo_cliente","canal"]].copy()
df_surv["tempo"] = np.minimum(df_surv["dias_desde_ultima"], CHURN_DIAS * 2)

def kaplan_meier(df_km, col_tempo="tempo", col_evento="churn", max_t=540, step=30):
    """KM simplificado sem lifelines."""
    tempos = np.arange(0, max_t + step, step)
    n_risk    = []
    n_events  = []
    survival  = []
    S = 1.0
    for t in tempos:
        at_risk  = (df_km[col_tempo] >= t).sum()
        events   = ((df_km[col_tempo] >= t) & (df_km[col_tempo] < t + step) & (df_km[col_evento] == 1)).sum()
        if at_risk > 0:
            h = events / at_risk
            S *= (1 - h)
        n_risk.append(at_risk)
        n_events.append(events)
        survival.append(S)
    return pd.DataFrame({"tempo": tempos, "sobrevivencia": survival,
                          "n_risk": n_risk, "n_events": n_events})

km_geral = kaplan_meier(df_surv)
print(f"\nCurva de sobrevivência (% clientes ainda ativos):")
for _, row in km_geral[km_geral["tempo"].isin([0,90,180,270,360,450,540])].iterrows():
    print(f"  {int(row['tempo']):>4} dias: {row['sobrevivencia']*100:.1f}%")

# KM por tipo de cliente
km_por_tipo = {}
for tipo in df_surv["tipo_cliente"].dropna().unique():
    km_por_tipo[tipo] = kaplan_meier(df_surv[df_surv["tipo_cliente"] == tipo])

# KM por canal
km_por_canal = {}
for canal in df_surv["canal"].dropna().unique():
    km_por_canal[canal] = kaplan_meier(df_surv[df_surv["canal"] == canal])

# ──────────────────────────────────────────
# 6. PREDIÇÃO DE CHURN
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("6. PREDIÇÃO DE RISCO DE CHURN")
print("=" * 60)

# Features para o modelo
features_base = ["n_pedidos","ticket_medio","ticket_total","dias_desde_primeira",
                 "dias_desde_ultima"]

df_model = clientes_elegíveis.copy()

# Encode categoricas
for col in ["tipo_cliente","canal","uf"]:
    if col in df_model.columns:
        le = LabelEncoder()
        df_model[col + "_enc"] = le.fit_transform(df_model[col].fillna("UNKNOWN"))
        features_base.append(col + "_enc")

# Adiciona features RFM se disponível
if TEM_RFM:
    df_model = df_model.merge(
        rfm[["cli_document","R","F","M","rfm_total"]], on="cli_document", how="left"
    )
    for col in ["R","F","M","rfm_total"]:
        if col in df_model.columns:
            features_base.append(col)

features = [f for f in features_base if f in df_model.columns]
df_model_clean = df_model.dropna(subset=features + ["churn"])

X = df_model_clean[features]
y = df_model_clean["churn"]

print(f"\nFeatures usadas: {features}")
print(f"Amostras: {len(X):,} | Churn rate: {y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.08,
    subsample=0.8, random_state=42
)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print(f"\nAUC-ROC: {auc:.3f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Ativo','Churned'])}")

# Importância das features
feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nImportância das features:\n{feat_imp.round(3)}")

# Score de risco para todos os clientes
df_model_clean = df_model_clean.copy()
df_model_clean["prob_churn"] = model.predict_proba(X[df_model_clean.index.isin(X.index)])[:, 1]

# Top clientes em risco (ainda ativos)
top_risco = (
    df_model_clean[df_model_clean["churn"] == 0]
    .sort_values("prob_churn", ascending=False)
    [["cli_document","prob_churn","dias_desde_ultima","n_pedidos","ticket_total"]]
    .head(10)
)
print(f"\nTop 10 clientes ATIVOS com maior risco de churn:")
print(top_risco.to_string(index=False))

# ──────────────────────────────────────────
# 7. VISUALIZAÇÕES
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("7. GERANDO VISUALIZAÇÕES")
print("=" * 60)

fig = plt.figure(figsize=(20, 24))
fig.suptitle(f"Análise de Churn — Dermage  |  Threshold: {CHURN_DIAS} dias",
             fontsize=17, fontweight="bold", color=COR_BASE, y=0.99)
gs = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── 7.1 Pizza churn geral ──
ax1 = fig.add_subplot(gs[0, 0])
sizes  = [n_churn, n_ativo]
labels = [f"Churned\n{taxa_churn:.1f}%", f"Ativo\n{100-taxa_churn:.1f}%"]
wedges, texts = ax1.pie(sizes, labels=labels, colors=[COR_CHURN, COR_ATIVO],
                         startangle=90, wedgeprops={"edgecolor":"white","linewidth":3})
for t in texts: t.set_fontsize(12); t.set_fontweight("bold")
ax1.set_title("Taxa de Churn Geral", fontweight="bold", fontsize=13)

# ── 7.2 Churn por segmento RFM ──
ax2 = fig.add_subplot(gs[0, 1])
if TEM_RFM:
    seg_plot = seg_churn.reset_index().sort_values("taxa_churn_%", ascending=True)
    colors_seg = [COR_CHURN if x > 50 else COR_ROXO if x > 30 else COR_ATIVO
                  for x in seg_plot["taxa_churn_%"]]
    bars = ax2.barh(seg_plot["segmento"], seg_plot["taxa_churn_%"],
                    color=colors_seg, edgecolor="white")
    ax2.set_title("Taxa de Churn por Segmento RFM", fontweight="bold")
    ax2.set_xlabel("% Churn")
    ax2.axvline(taxa_churn, color="#AAAAAA", linestyle="--", linewidth=1.5,
                label=f"Média: {taxa_churn:.1f}%")
    ax2.legend(fontsize=9)
    for bar in bars:
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.1f}%", va="center", fontsize=10, fontweight="bold")
else:
    ax2.text(0.5, 0.5, "RFM não disponível", ha="center", va="center",
             transform=ax2.transAxes, color="#AAAAAA")

# ── 7.3 Curva de sobrevivência geral ──
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(km_geral["tempo"], km_geral["sobrevivencia"] * 100,
         color=COR_BASE, linewidth=3, label="Geral")
# Por tipo de cliente
cores_tipo = [COR_CHURN, COR_ATIVO]
for (tipo, km_t), cor in zip(km_por_tipo.items(), cores_tipo):
    ax3.plot(km_t["tempo"], km_t["sobrevivencia"] * 100,
             linewidth=2, linestyle="--", color=cor, label=tipo, alpha=0.85)
# Linhas de referência
for ref, label in [(180, "180d"), (365, "365d")]:
    ax3.axvline(ref, color="#CCCCCC", linestyle=":", linewidth=1.5)
    ax3.text(ref + 5, 95, label, fontsize=9, color="#AAAAAA")
ax3.fill_between(km_geral["tempo"], km_geral["sobrevivencia"] * 100, alpha=0.08, color=COR_BASE)
ax3.set_title("Curva de Sobrevivência — % Clientes Ainda Ativos ao Longo do Tempo",
              fontweight="bold", fontsize=13)
ax3.set_xlabel("Dias desde a última compra")
ax3.set_ylabel("% Clientes sem churn")
ax3.set_ylim(0, 105)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax3.legend(title="Segmento", fontsize=10)

# ── 7.4 Importância das features ──
ax4 = fig.add_subplot(gs[2, 0])
feat_plot = feat_imp.head(10).sort_values()
colors_feat = [COR_CHURN if i >= len(feat_plot)-3 else COR_ROXO
               for i in range(len(feat_plot))]
feat_plot.plot(kind="barh", ax=ax4, color=colors_feat, edgecolor="white")
ax4.set_title(f"Importância das Features\n(AUC-ROC: {auc:.3f})", fontweight="bold")
ax4.set_xlabel("Importância")

# ── 7.5 Distribuição de probabilidade de churn ──
ax5 = fig.add_subplot(gs[2, 1])
churned_proba = df_model_clean[df_model_clean["churn"] == 1]["prob_churn"]
ativo_proba   = df_model_clean[df_model_clean["churn"] == 0]["prob_churn"]
ax5.hist(ativo_proba,   bins=40, color=COR_ATIVO,  alpha=0.6, label="Ativo",   density=True)
ax5.hist(churned_proba, bins=40, color=COR_CHURN,  alpha=0.6, label="Churned", density=True)
ax5.set_title("Distribuição de P(Churn) pelo Modelo", fontweight="bold")
ax5.set_xlabel("Probabilidade de Churn")
ax5.set_ylabel("Densidade")
ax5.legend()

# ── 7.6 Curva ROC ──
ax6 = fig.add_subplot(gs[3, 0])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax6.plot(fpr, tpr, color=COR_CHURN, linewidth=2.5, label=f"AUC = {auc:.3f}")
ax6.plot([0,1],[0,1], color="#CCCCCC", linestyle="--", linewidth=1)
ax6.fill_between(fpr, tpr, alpha=0.1, color=COR_CHURN)
ax6.set_title("Curva ROC — Modelo de Predição de Churn", fontweight="bold")
ax6.set_xlabel("Taxa de Falsos Positivos")
ax6.set_ylabel("Taxa de Verdadeiros Positivos")
ax6.legend(fontsize=11)

# ── 7.7 Churn por UF ──
ax7 = fig.add_subplot(gs[3, 1])
uf_plot = uf_churn.head(10).sort_values("mean").reset_index()
colors_uf = [COR_CHURN if x > taxa_churn/100 else COR_ATIVO for x in uf_plot["mean"]]
ax7.barh(uf_plot["uf"], uf_plot["mean"] * 100, color=colors_uf, edgecolor="white")
ax7.axvline(taxa_churn, color="#AAAAAA", linestyle="--", linewidth=1.5,
            label=f"Média: {taxa_churn:.1f}%")
ax7.set_title("Taxa de Churn por UF (top 10, mín. 100 clientes)", fontweight="bold")
ax7.set_xlabel("% Churn")
ax7.legend(fontsize=9)

plt.savefig("churn_dermage.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\n✅ Gráfico salvo: churn_dermage.png")
plt.show()

# ──────────────────────────────────────────
# 8. EXPORT
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("8. EXPORTANDO")
print("=" * 60)

df_model_clean[["cli_document","churn","prob_churn","dias_desde_ultima",
                "n_pedidos","ticket_total","tipo_cliente","canal","uf"]].to_csv(
    "churn_scores.csv", index=False)
print("✅ churn_scores.csv — score de risco para todos os clientes elegíveis")

# ──────────────────────────────────────────
# 9. RESUMO EXECUTIVO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("9. RESUMO EXECUTIVO — CHURN")
print("=" * 60)

print(f"""
🚨 CHURN — DERMAGE  (threshold = {CHURN_DIAS} dias)

📊 TAXA GERAL
   • {taxa_churn:.1f}% dos clientes elegíveis churnou
   • {n_churn:,} clientes perdidos | {n_ativo:,} ainda ativos

👤 PERFIL DO CLIENTE CHURNED vs ATIVO
   • Ticket médio churned: R${perfil.loc['Churned','ticket_medio']:.2f}
   • Ticket médio ativo:   R${perfil.loc['Ativo','ticket_medio']:.2f}
   • Pedidos médios churned: {perfil.loc['Churned','n_pedidos_med']:.1f}
   • Pedidos médios ativo:   {perfil.loc['Ativo','n_pedidos_med']:.1f}

🤖 MODELO DE PREDIÇÃO
   • AUC-ROC: {auc:.3f} ({'ótimo' if auc > 0.80 else 'bom' if auc > 0.70 else 'razoável'})
   • Feature mais importante: {feat_imp.index[0]}
   • 2ª feature: {feat_imp.index[1]}

💡 AÇÃO RECOMENDADA
   • Clientes com prob_churn > 0.7: campanha de reativação urgente
   • Clientes com prob_churn 0.4–0.7: nurturing com oferta de recompra
   • Clientes com prob_churn < 0.4: manter programa de fidelidade
""")