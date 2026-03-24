"""
MERGE — Dermage Datathon Girls in Tech 2026
Cruzamento entre base_pedidos.csv e base_produtos.csv
1 pedido pode ter N produtos → cuidado com agregações!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
ARQUIVO_PEDIDOS  = "base_pedidos.csv"
ARQUIVO_PRODUTOS = "base_produtos.csv"

COR_1    = "#E75480"
COR_ROSA = "#E75480"
COR_2    = "#6A5ACD"
COR_3    = "#2E8B57"
COR_BASE = "#2E4057"
COR_ROXO = "#6A5ACD"
COR_VERDE = "#2E8B57"
PALETTE  = [COR_1, COR_2, COR_3, "#F4A261", "#264653", "#E9C46A", "#A8DADC"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})

# ──────────────────────────────────────────
# 1. CARREGA E LIMPA PEDIDOS
# ──────────────────────────────────────────
print("=" * 60)
print("1. CARREGANDO BASE DE PEDIDOS")
print("=" * 60)

df_ped = pd.read_csv(ARQUIVO_PEDIDOS, encoding="utf-8-sig", decimal=",", dtype=str,sep=';')

df_ped["value"] = (
    df_ped["value"].astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)
for col in ["data", "data_tratada"]:
    if col in df_ped.columns:
        df_ped[col] = pd.to_datetime(df_ped[col], dayfirst=True, errors="coerce")

# Filtra válidos
df_ped_val = df_ped[df_ped["status"] == "invoiced"].copy()
df_ped_val.rename(columns={
    "orderid":       "order_id",
    "value":         "valor_pedido",
    "cli_document":  "cli_document",
    "tipo_cliente":  "tipo_cliente",
    "data":          "data_pedido",
}, inplace=True)

print(f"Pedidos válidos: {len(df_ped_val):,}")
print(f"Colunas: {list(df_ped_val.columns)}")

# ──────────────────────────────────────────
# 2. CARREGA E LIMPA PRODUTOS
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CARREGANDO BASE DE PRODUTOS")
print("=" * 60)

df_pro = pd.read_csv(ARQUIVO_PRODUTOS, encoding="utf-8-sig", decimal=".", dtype=str,sep=';')

rename_pro = {
    "Origin": "canal", "Order": "order_id", "Creation Date": "data_produto",
    "Client Document": "cli_document_pro", "UF": "uf", "Status": "status_pro",
    "Payment System Name": "pagamento", "Installments": "parcelas",
    "Payment Value": "valor_pago", "Quantity_SKU": "qtd_sku",
    "ID_SKU": "id_sku", "Reference Code": "ref_code", "SKU Name": "sku_name",
    "SKU Value": "sku_value", "SKU Selling Price": "sku_preco_venda",
    "SKU Total Price": "sku_total", "Total Value": "total_pedido",
    "Discounts Totals": "desconto", "Seller Name": "seller",
}
df_pro.rename(columns={k: v for k, v in rename_pro.items() if k in df_pro.columns}, inplace=True)

cols_num = ["valor_pago","qtd_sku","sku_value","sku_preco_venda",
            "sku_total","total_pedido","desconto","parcelas"]
for c in cols_num:
    if c in df_pro.columns:
        df_pro[c] = pd.to_numeric(df_pro[c], errors="coerce")

df_pro["data_produto"] = pd.to_datetime(df_pro["data_produto"], errors="coerce")

STATUS_PRO_VALIDO = ["Faturado", "invoiced", "Faturada"]
df_pro_val = df_pro[df_pro["status_pro"].isin(STATUS_PRO_VALIDO)].copy()

# Extrai família
# ── Famílias oficiais Dermage (familias.csv) ──
FAMILIAS_OFICIAIS = [
    "Photoage", "Secatriz", "Hyaluage", "Improve", "Revicare",
    "Compose", "Glycolique", "Revox", "Vinocare", "Clarité",
    "Revitrat", "Ineout", "EXOCARE", "Age Inverse",
]

def extrair_familia(nome):
    if pd.isna(nome):
        return "Outros"
    nome_clean = nome.strip()
    for f in FAMILIAS_OFICIAIS:
        if nome_clean.lower().startswith(f.lower()):
            return f
    return "Outros"

df_pro_val["familia"] = df_pro_val["sku_name"].apply(extrair_familia)

print(f"Linhas de produto válidas: {len(df_pro_val):,}")
print(f"Pedidos únicos em produtos: {df_pro_val['order_id'].nunique():,}")

# ──────────────────────────────────────────
# 3. MERGE
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MERGE: PEDIDOS ← PRODUTOS")
print("=" * 60)

# ⚠️ LEFT JOIN: pedidos é a base principal.
# Cada linha do merge = 1 SKU dentro de 1 pedido.
# Colunas de valor do pedido (valor_pedido) NÃO devem ser somadas aqui —
# use sku_total para somar receita por produto.

df_merge = df_ped_val.merge(
    df_pro_val,
    on="order_id",
    how="left",
    suffixes=("_ped", "_pro")
)

print(f"\nShape após merge: {df_merge.shape}")
print(f"Pedidos com pelo menos 1 produto: {df_merge['order_id'].nunique():,}")
print(f"Pedidos sem match em produtos:    {df_ped_val[~df_ped_val['order_id'].isin(df_pro_val['order_id'])].shape[0]:,}")

# Diagnóstico de consistência de valor
df_merge["diff_valor"] = (df_merge["valor_pedido"] - df_merge["total_pedido"]).abs()
sem_diff = (df_merge["diff_valor"] < 1).mean() * 100
print(f"\nPedidos com valor consistente (diff < R$1): {sem_diff:.1f}%")

# ──────────────────────────────────────────
# 4. VISÃO PEDIDO-NÍVEL (sem dupla contagem)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("4. VISÃO PEDIDO-NÍVEL (agregado por order_id)")
print("=" * 60)

# Agrega informações de produto por pedido
agg_pedido = df_merge.groupby("order_id").agg(
    tipo_cliente   = ("tipo_cliente",   "first"),
    data_pedido    = ("data_pedido",    "first"),
    valor_pedido   = ("valor_pedido",   "first"),   # valor real do pedido (não somar)
    canal          = ("canal",          "first"),
    uf             = ("uf",             "first"),
    pagamento      = ("pagamento",      "first"),
    parcelas       = ("parcelas",       "first"),
    n_skus         = ("id_sku",         "nunique"), # quantos SKUs distintos
    qtd_total_itens= ("qtd_sku",        "sum"),     # total de unidades
    familias       = ("familia",        lambda x: sorted(set(x.dropna()))),
    n_familias     = ("familia",        "nunique"),
    receita_sku    = ("sku_total",      "sum"),     # soma dos SKUs (pode diferir do total por frete)
    desconto_total = ("desconto",       "sum"),
).reset_index()

# Tamanho do carrinho
print(f"\nEstatísticas de SKUs por pedido:")
print(agg_pedido["n_skus"].describe().round(2))

print(f"\nDistribuição de tamanho do carrinho:")
print(agg_pedido["n_skus"].value_counts().sort_index().head(10))

# Pedidos com mais de 1 família
multi_fam = agg_pedido[agg_pedido["n_familias"] > 1]
print(f"\nPedidos com produtos de múltiplas famílias: {len(multi_fam):,} ({len(multi_fam)/len(agg_pedido)*100:.1f}%)")

# ──────────────────────────────────────────
# 5. RECOMPRA ENRIQUECIDA
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. RECOMPRA ENRIQUECIDA COM PRODUTOS")
print("=" * 60)

# Ordena compras por cliente
df_cli = agg_pedido.sort_values(["tipo_cliente","data_pedido"]).copy()
# Número de compra por cliente (usando cli_document do merge)
cli_map = df_merge[["order_id","cli_document"]].drop_duplicates().set_index("order_id")["cli_document"]
df_cli["cli_document"] = df_cli["order_id"].map(cli_map)

df_cli = df_cli.sort_values(["cli_document","data_pedido"])
df_cli["n_compra"] = df_cli.groupby("cli_document").cumcount() + 1

# Família na 1ª compra
fam_1a = (
    df_merge.merge(df_cli[df_cli["n_compra"]==1][["order_id"]], on="order_id")
    .groupby("familia")["order_id"].nunique()
    .sort_values(ascending=False)
)
fam_2a = (
    df_merge.merge(df_cli[df_cli["n_compra"]==2][["order_id"]], on="order_id")
    .groupby("familia")["order_id"].nunique()
    .sort_values(ascending=False)
)

print(f"\nTop famílias na 1ª compra:")
print(fam_1a.head(8))
print(f"\nTop famílias na 2ª compra:")
print(fam_2a.head(8))

# Ticket médio por n_compra (até 5ª)
ticket_por_compra = df_cli[df_cli["n_compra"] <= 5].groupby("n_compra")["valor_pedido"].mean()
print(f"\nTicket médio por número da compra:")
print(ticket_por_compra.round(2))

# ──────────────────────────────────────────
# 6. SEGMENTAÇÃO RFM SIMPLES
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SEGMENTAÇÃO RFM")
print("=" * 60)

data_ref = df_cli["data_pedido"].max()

rfm = df_cli.groupby("cli_document").agg(
    recencia    = ("data_pedido",  lambda x: (data_ref - x.max()).days),
    frequencia  = ("order_id",     "count"),
    monetario   = ("valor_pedido", "sum"),
).reset_index()

# Score RFM (quintis 1-5) via rank — robusto a duplicatas
rfm["R"] = pd.cut(
    rfm["recencia"].rank(method="first", ascending=True),
    bins=5, labels=[5, 4, 3, 2, 1]   # mais recente → score 5
).astype(int)

rfm["F"] = pd.cut(
    rfm["frequencia"].rank(method="first", ascending=False),
    bins=5, labels=[5, 4, 3, 2, 1]   # mais frequente → score 5
).astype(int)

rfm["M"] = pd.cut(
    rfm["monetario"].rank(method="first", ascending=False),
    bins=5, labels=[5, 4, 3, 2, 1]   # maior gasto → score 5
).astype(int)

rfm["rfm_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
rfm["rfm_total"] = rfm["R"] + rfm["F"] + rfm["M"]


# Segmentos simplificados
def segmento(row):
    # Skin Glow: alta recência E alta frequência → clientes fiéis e ativos
    if row["R"] >= 3 and row["F"] >= 3:
        return "Skin Glow"
    # Skin SOS: baixa recência E baixa frequência → dormentes/perdidos
    if row["R"] <= 2 and row["F"] <= 2:
        return "Skin SOS"
    # Skin Bloom: todo o restante → em transição, podem ir para qualquer lado
    return "Skin Bloom"
rfm["segmento"] = rfm.apply(segmento, axis=1)

print(f"\nDistribuição de segmentos RFM:")
seg_dist = rfm["segmento"].value_counts()
print(seg_dist)

print(f"\nKPIs por segmento:")
print(rfm.groupby("segmento").agg(
    clientes   = ("cli_document", "count"),
    rec_media  = ("recencia",     "mean"),
    freq_media = ("frequencia",   "mean"),
    ticket_med = ("monetario",    "mean"),
).round(1).sort_values("clientes", ascending=False))

# ──────────────────────────────────────────
# 7. CANAL × RECOMPRA
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("7. CANAL × TIPO DE CLIENTE")
print("=" * 60)

canal_tipo = agg_pedido.groupby(["canal","tipo_cliente"])["order_id"].count().unstack(fill_value=0)
print(f"\n{canal_tipo}")

canal_ticket = agg_pedido.groupby(["canal","tipo_cliente"])["valor_pedido"].mean().unstack().round(2)
print(f"\nTicket médio por canal e tipo:\n{canal_ticket}")

# ──────────────────────────────────────────
# 8. VISUALIZAÇÕES DO MERGE
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GERANDO VISUALIZAÇÕES...")
print("=" * 60)

fig = plt.figure(figsize=(20, 24))
fig.suptitle("Merge Pedidos × Produtos — Dermage", fontsize=18,
             fontweight="bold", color=COR_BASE, y=0.99)
gs = GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

# ── 8.1 Tamanho do carrinho ──
ax1 = fig.add_subplot(gs[0, 0])
cart = agg_pedido["n_skus"].value_counts().sort_index().head(8)
ax1.bar(cart.index.astype(str), cart.values, color=COR_2, edgecolor="white")
ax1.set_title("Tamanho do Carrinho (SKUs por Pedido)", fontweight="bold")
ax1.set_xlabel("Nº de SKUs distintos")
ax1.set_ylabel("Pedidos")
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height()):,}",
                 (p.get_x() + p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=9)

# ── 8.2 Ticket por nº de compra ──
ax2 = fig.add_subplot(gs[0, 1])
tpc = ticket_por_compra.reset_index()
ax2.bar(tpc["n_compra"].astype(str), tpc["valor_pedido"],
        color=[COR_1 if i == 0 else COR_2 for i in range(len(tpc))],
        edgecolor="white")
ax2.set_title("Ticket Médio por Número da Compra", fontweight="bold")
ax2.set_xlabel("Nº da compra")
ax2.set_ylabel("R$")
for p in ax2.patches:
    ax2.annotate(f"R${p.get_height():.0f}",
                 (p.get_x() + p.get_width()/2, p.get_height() + 1),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

# ── 8.3 1ª vs 2ª compra por família ──
ax3 = fig.add_subplot(gs[1, 0])
all_fams = list(dict.fromkeys(list(fam_1a.head(7).index) + list(fam_2a.head(7).index)))[:8]
x = np.arange(len(all_fams))
w = 0.35
ax3.bar(x - w/2, [fam_1a.get(f, 0) for f in all_fams], w, color=COR_1, label="1ª compra", edgecolor="white")
ax3.bar(x + w/2, [fam_2a.get(f, 0) for f in all_fams], w, color=COR_2, label="2ª compra", edgecolor="white")
ax3.set_xticks(x)
ax3.set_xticklabels(all_fams, rotation=35, ha="right", fontsize=9)
ax3.set_title("Top Famílias: 1ª vs 2ª Compra", fontweight="bold")
ax3.set_ylabel("Pedidos")
ax3.legend()

# ── 8.4 Segmentação RFM ──
ax4 = fig.add_subplot(gs[1, 1])
seg_plot = rfm["segmento"].value_counts()
wedges, texts, autotexts = ax4.pie(
    seg_plot.values, labels=seg_plot.index,
    colors=PALETTE[:len(seg_plot)],
    autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor":"white","linewidth":2}
)
for at in autotexts: at.set_fontsize(9)
ax4.set_title("Segmentação RFM de Clientes", fontweight="bold")

# ── 8.5 Canal × tipo de cliente ──
ax5 = fig.add_subplot(gs[2, 0])
ct = canal_tipo.reset_index()
x = np.arange(len(ct))
w = 0.35
for i, (col, cor) in enumerate(zip(ct.columns[1:], [COR_1, COR_2])):
    ax5.bar(x + (i-0.5)*w, ct[col], w, label=col, color=cor, edgecolor="white")
ax5.set_xticks(x)
ax5.set_xticklabels(ct["canal"], rotation=0)
ax5.set_title("Canal × Tipo de Cliente", fontweight="bold")
ax5.set_ylabel("Pedidos")
ax5.legend(title="Tipo")

# ── 8.6 Ticket médio por canal e tipo ──
ax6 = fig.add_subplot(gs[2, 1])
ct2 = canal_ticket.reset_index()
x = np.arange(len(ct2))
for i, (col, cor) in enumerate(zip(ct2.columns[1:], [COR_1, COR_2])):
    bars = ax6.bar(x + (i-0.5)*w, ct2[col], w, label=col, color=cor, edgecolor="white")
    for bar in bars:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"R${bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
ax6.set_xticks(x)
ax6.set_xticklabels(ct2["canal"], rotation=0)
ax6.set_title("Ticket Médio: Canal × Tipo de Cliente", fontweight="bold")
ax6.set_ylabel("R$")
ax6.legend(title="Tipo")

# ── 8.7 Recência × Frequência (RFM scatter) ──
ax7 = fig.add_subplot(gs[3, :])
seg_colors = {
    "Campeões": COR_1, "Leais": COR_2, "Novos Promissores": COR_3,
    "Em Risco": "#F4A261", "Hibernando": "#AAAAAA", "Potencial": COR_BASE
}
for seg, grp in rfm.groupby("segmento"):
    ax7.scatter(grp["recencia"], grp["frequencia"],
                alpha=0.4, s=grp["monetario"]/grp["monetario"].max()*200 + 10,
                color=seg_colors.get(seg, "#999999"), label=seg)
ax7.set_title("Mapa RFM: Recência × Frequência (tamanho = valor monetário)",
              fontweight="bold")
ax7.set_xlabel("Recência (dias desde última compra)")
ax7.set_ylabel("Frequência (nº de compras)")
ax7.legend(title="Segmento", bbox_to_anchor=(1.01, 1), loc="upper left")

plt.savefig("eda_dermage_merge.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\n✅ Gráfico salvo: eda_dermage_merge.png")
plt.show()

# ──────────────────────────────────────────
# 9. EXPORTS ÚTEIS
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("9. EXPORTANDO TABELAS ANALÍTICAS")
print("=" * 60)

# df_merge: 1 linha por SKU dentro do pedido (base de trabalho principal)
df_merge.to_csv("dermage_merge_sku.csv", index=False)
print("✅ dermage_merge_sku.csv  — 1 linha por SKU × pedido")

# agg_pedido: 1 linha por pedido com infos agregadas
agg_pedido.to_csv("dermage_pedido_agregado.csv", index=False)
print("✅ dermage_pedido_agregado.csv — 1 linha por pedido")

# rfm: 1 linha por cliente com scores e segmento
rfm.to_csv("dermage_rfm_clientes.csv", index=False)
print("✅ dermage_rfm_clientes.csv — 1 linha por cliente (RFM)")

print(f"""
📊 RESUMO DO MERGE
   • Linhas do merge (1 SKU/linha): {len(df_merge):,}
   • Pedidos únicos: {df_merge['order_id'].nunique():,}
   • Clientes únicos: {rfm.shape[0]:,}
   • Carrinho médio: {agg_pedido['n_skus'].mean():.1f} SKUs / pedido
   • Pedidos multi-família: {len(multi_fam):,} ({len(multi_fam)/len(agg_pedido)*100:.1f}%)
""")

############ ANÁLISE DE COHORT ######################################################


# 1. Identifica mês da 1ª compra de cada cliente (coorte)
primeira_compra = (
    df_merge
    .groupby("cli_document")["data_pedido"]
    .min()
    .dt.to_period("M")
    .rename("coorte")
)

# 2. Mês de cada compra
df_cohort = df_merge.merge(primeira_compra, on="cli_document")
df_cohort["periodo"] = (
    df_cohort["data_pedido"].dt.to_period("M") - df_cohort["coorte"]
).apply(lambda x: x.n)  # distância em meses

# 3. Conta clientes únicos por coorte e período
cohort_data = (
    df_cohort
    .groupby(["coorte", "periodo"])["cli_document"]
    .nunique()
    .reset_index()
)

# 4. Pivota para a matriz
cohort_matrix = cohort_data.pivot_table(
    index="coorte", columns="periodo", values="cli_document"
)

# 5. Converte para % de retenção
cohort_pct = cohort_matrix.divide(cohort_matrix[0], axis=0).round(3) * 100


#######################################################

"""
HEATMAP DE COORTE — Dermage Datathon Girls in Tech 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
USAR_MERGE_CSV = True
ARQUIVO_MERGE  = "dermage_merge_sku.csv"
ARQUIVO_PEDIDOS  = "base_pedidos.csv"
ARQUIVO_PRODUTOS = "base_produtos.csv"

cmap_cohort = LinearSegmentedColormap.from_list(
    "dermage", ["#FFFFFF", "#F4B8C8", "#E75480", "#8B0000"], N=256
)

# ──────────────────────────────────────────
# 1. CARREGAMENTO
# ──────────────────────────────────────────
if USAR_MERGE_CSV:
    df = pd.read_csv(ARQUIVO_MERGE, low_memory=False)
    df["data_pedido"] = pd.to_datetime(df["data_pedido"], errors="coerce")
else:
    df_ped = pd.read_csv(ARQUIVO_PEDIDOS, encoding="utf-8-sig", decimal=",", dtype=str)
    df_ped["value"] = (df_ped["value"].str.replace(".", "", regex=False)
                                       .str.replace(",", ".", regex=False).astype(float))
    df_ped["data"]  = pd.to_datetime(df_ped["data"], dayfirst=True, errors="coerce")
    df_ped_val = df_ped[df_ped["status"] == "invoiced"].copy()
    df_ped_val.rename(columns={"orderid":"order_id","value":"valor_pedido","data":"data_pedido"}, inplace=True)

    df_pro = pd.read_csv(ARQUIVO_PRODUTOS, encoding="utf-8-sig", decimal=".", dtype=str)
    df_pro.rename(columns={"Order":"order_id","Client Document":"cli_document"}, inplace=True)
    STATUS_OK = ["Faturado","invoiced","Faturada"]
    df_pro_val = df_pro[df_pro.get("Status", df_pro.get("status_pro","")).isin(STATUS_OK)]
    df = df_ped_val.merge(df_pro_val[["order_id","cli_document"]].drop_duplicates("order_id"),
                          on="order_id", how="left")

if "cli_document_x" in df.columns:
    df["cli_document"] = df["cli_document_x"].fillna(df.get("cli_document_y", ""))

# ──────────────────────────────────────────
# 2. MONTA MATRIZ DE COORTE
# ──────────────────────────────────────────
df_ped_unico = (
    df.dropna(subset=["cli_document","data_pedido"])
      .drop_duplicates(subset=["cli_document","order_id"])
      [["cli_document","order_id","data_pedido"]]
      .copy()
)

primeira_compra = (
    df_ped_unico.groupby("cli_document")["data_pedido"]
    .min().dt.to_period("M").rename("coorte").reset_index()
)

df_cohort = df_ped_unico.merge(primeira_compra, on="cli_document")
df_cohort["periodo"] = (
    df_cohort["data_pedido"].dt.to_period("M") - df_cohort["coorte"]
).apply(lambda x: x.n)

MAX_PERIODO = 17
cohort_counts = (
    df_cohort[df_cohort["periodo"] <= MAX_PERIODO]
    .groupby(["coorte","periodo"])["cli_document"]
    .nunique().reset_index()
)

matrix = cohort_counts.pivot_table(index="coorte", columns="periodo", values="cli_document")
pct    = matrix.divide(matrix[0], axis=0).round(4) * 100

# ──────────────────────────────────────────
# 3. HEATMAP
# ──────────────────────────────────────────
n_coortes  = len(pct)
n_periodos = len(pct.columns)
fig_h = max(8, n_coortes * 0.52)
fig_w = max(16, n_periodos * 0.85)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")

sns.heatmap(
    pct,
    ax=ax,
    cmap=cmap_cohort,
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    linecolor="#EEEEEE",
    cbar_kws={"label": "% Retenção", "shrink": 0.5, "pad": 0.02},
    vmin=0, vmax=35,
    yticklabels=[str(c) for c in pct.index],
    annot_kws={"size": 9, "weight": "bold"},
)

# Tamanho de cada coorte (n=) à direita
for i, (idx, row) in enumerate(matrix.iterrows()):
    ax.text(
        pct.shape[1] + 0.25, i + 0.5,
        f"n={int(row[0]):,}",
        va="center", ha="left", fontsize=9, color="#444444"
    )

ax.set_title(
    "Retenção de Clientes por Coorte — Dermage",
    fontsize=15, fontweight="bold", color="#2E4057", pad=16
)
ax.set_xlabel("Meses após 1ª compra", fontsize=11, labelpad=8)
ax.set_ylabel("Coorte (mês da 1ª compra)", fontsize=11, labelpad=8)
ax.tick_params(axis="x", rotation=0, labelsize=10)
ax.tick_params(axis="y", rotation=0, labelsize=10)

# Linha separadora após mês 0
ax.axvline(x=1, color="#E75480", linewidth=2, alpha=0.6)

plt.tight_layout()
plt.savefig("cohort_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✅ Salvo: cohort_heatmap.png")
plt.show()



############ MRR — CENÁRIO BASEADO EM RECOMPRA REAL ##########################

# ── Premissas baseadas nos dados reais ──
TAXA_RECOMPRA_REAL   = 0.235   # 23,5% dos clientes fazem 2ª compra
INCREMENTO_RETENCAO  = 0.25    # cenário: +25% sobre a taxa de recompra
TAXA_RECOMPRA_CEN    = TAXA_RECOMPRA_REAL * (1 + INCREMENTO_RETENCAO)  # → 29,4%
TICKET_RECOMPRA      = 242.09  # ticket médio da 2ª compra (real)
JANELA_RECOMPRA_DIAS = 117     # mediana de dias até 2ª compra

print(f"""
📐 PREMISSAS DO CENÁRIO
─────────────────────────────────────────
   Taxa de recompra real:      {TAXA_RECOMPRA_REAL*100:.1f}%
   Taxa de recompra cenário:   {TAXA_RECOMPRA_CEN*100:.1f}% (+{INCREMENTO_RETENCAO*100:.0f}%)
   Ticket médio da recompra:   R$ {TICKET_RECOMPRA:.2f}
   Janela de recompra:         {JANELA_RECOMPRA_DIAS} dias (~{JANELA_RECOMPRA_DIAS//30} meses)
─────────────────────────────────────────
""")

# ── Base: 1 linha por pedido ──
df_mrr = (
    df_merge
    .dropna(subset=["cli_document", "data_pedido", "tipo_cliente"])
    .drop_duplicates(subset=["cli_document", "order_id"])
    [["cli_document", "order_id", "data_pedido", "valor_pedido", "tipo_cliente"]]
    .copy()
)
df_mrr["mes"] = df_mrr["data_pedido"].dt.to_period("M")

# ── Novos clientes por mês (eles são os elegíveis para recompra futura) ──
# Um cliente NOVO num mês X é elegível para recomprar ~4 meses depois
JANELA_MESES = round(JANELA_RECOMPRA_DIAS / 30)  # 4 meses

novos_por_mes = (
    df_mrr[df_mrr["tipo_cliente"] == "NOVO"]
    .groupby("mes")["cli_document"]
    .nunique()
    .reset_index()
    .rename(columns={"cli_document": "novos"})
)

# ── MRR real por mês ──
mensal = (
    df_mrr.groupby(["mes", "tipo_cliente"])["valor_pedido"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)
mensal["mes_str"]  = mensal["mes"].astype(str)
mensal["mrr_real"] = mensal.get("NOVO", 0) + mensal.get("ANTIGO", 0)

# ── Receita incremental por mês ──
# Lógica: novos clientes do mês X retornam no mês X + JANELA_MESES
# Incremental = novos[mês X] × (taxa_cenario - taxa_real) × ticket_recompra
mensal = mensal.merge(novos_por_mes, on="mes", how="left")
mensal["novos"] = mensal["novos"].fillna(0)

# Shifta os novos clientes pela janela de recompra
mensal["novos_elegíveis"] = mensal["novos"].shift(JANELA_MESES, fill_value=0)

# Receita incremental = clientes adicionais que recompram × ticket
mensal["recompras_adicionais"] = mensal["novos_elegíveis"] * (TAXA_RECOMPRA_CEN - TAXA_RECOMPRA_REAL)
mensal["receita_incremental"]  = (mensal["recompras_adicionais"] * TICKET_RECOMPRA).round(2)
mensal["mrr_cenario"]          = mensal["mrr_real"] + mensal["receita_incremental"]

# ── Métricas dos cards ──
receita_real    = mensal["mrr_real"].sum()
receita_cenario = mensal["mrr_cenario"].sum()
ganho_total     = receita_cenario - receita_real
ganho_mes_medio = mensal["receita_incremental"].mean()

print(f"""
💳 CARDS DO GRÁFICO MRR
─────────────────────────────────────────
   Receita real (18 meses):      R$ {receita_real:,.0f}
   Receita cenário (+25% ret.):  R$ {receita_cenario:,.0f}
   Ganho incremental total:      R$ {ganho_total:,.0f}
   Ganho médio por mês:          R$ {ganho_mes_medio:,.0f}
   Recompras adicionais/mês:     {mensal['recompras_adicionais'].mean():.0f} clientes
─────────────────────────────────────────
""")

# ── Plot ──
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("white")

x     = np.arange(len(mensal))
width = 0.6

ax.bar(x, mensal.get("ANTIGO", 0),
       width=width, color=COR_ROXO, label="Clientes Antigos (real)",
       edgecolor="white", zorder=3)
ax.bar(x, mensal.get("NOVO", 0),
       width=width, bottom=mensal.get("ANTIGO", 0),
       color=COR_ROSA, label="Clientes Novos (real)",
       edgecolor="white", zorder=3)
ax.bar(x, mensal["receita_incremental"],
       width=width, bottom=mensal["mrr_real"],
       color=COR_VERDE, label=f"Recompras adicionais (+{int(INCREMENTO_RETENCAO*100)}% retenção)",
       edgecolor="white", alpha=0.8, zorder=3)

# Linha do cenário
ax.plot(x, mensal["mrr_cenario"], color=COR_BASE, linewidth=2.5,
        marker="o", markersize=5, label="Total cenário", zorder=5)

# Anotação com o ganho total
ax.annotate(
    f"Ganho total: R${ganho_total/1e6:.2f}M\n"
    f"({JANELA_MESES} meses após entrada do cliente\n"
    f"ticket recompra: R${TICKET_RECOMPRA:.0f})",
    xy=(x[-4], mensal["mrr_cenario"].iloc[-4]),
    xytext=(x[-9], mensal["mrr_cenario"].max() * 0.75),
    fontsize=9, color=COR_VERDE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COR_VERDE, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor=COR_VERDE, alpha=0.9)
)

ax.set_xticks(x)
ax.set_xticklabels(mensal["mes_str"], rotation=45, ha="right", fontsize=9)
ax.set_title(
    f"MRR Real vs Cenário +{int(INCREMENTO_RETENCAO*100)}% Retenção\n"
    f"(recompra em ~{JANELA_MESES} meses | ticket R${TICKET_RECOMPRA:.0f})",
    fontsize=13, fontweight="bold", color=COR_BASE, pad=14
)
ax.set_ylabel("Receita Mensal (R$)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"R${v/1000:.0f}k"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("mrr_cenario.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✅ Salvo: mrr_cenario.png")
plt.show()

##############################
############ DIAS DE RECOMPRA POR FAMÍLIA ####################################

# ── Base: ordena compras por cliente e data ──
df_fam = (
    df_merge
    .dropna(subset=["cli_document", "data_pedido", "familia"])
    .drop_duplicates(subset=["cli_document", "order_id"])
    [["cli_document", "order_id", "data_pedido", "familia"]]
    .copy()
    .sort_values(["cli_document", "data_pedido"])
)

# Família predominante do pedido (moda dos SKUs no pedido)
familia_pedido = (
    df_merge.dropna(subset=["order_id","familia"])
    .groupby("order_id")["familia"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"familia": "familia_pedido"})
)

df_fam = df_fam.merge(familia_pedido, on="order_id", how="left")

# Próxima compra do cliente
df_fam["data_proxima"] = df_fam.groupby("cli_document")["data_pedido"].shift(-1)
df_fam["dias_recompra"] = (df_fam["data_proxima"] - df_fam["data_pedido"]).dt.days

# Remove linhas sem recompra e outliers extremos
df_recompra = df_fam.dropna(subset=["dias_recompra"]).copy()
df_recompra = df_recompra[
    (df_recompra["dias_recompra"] > 0) &
    (df_recompra["dias_recompra"] <= 365)
]

# ── Estatísticas por família ──
MIN_OBS = 100   # mínimo de observações para incluir a família

stats_familia = (
    df_recompra.groupby("familia_pedido")["dias_recompra"]
    .agg(
        obs         = "count",
        media       = "mean",
        mediana     = "median",
        p25         = lambda x: x.quantile(0.25),
        p75         = lambda x: x.quantile(0.75),
        timing_ideal= lambda x: x.quantile(0.25),  # P25 = janela de ouro
    )
    .round(1)
    .query(f"obs >= {MIN_OBS}")
    .sort_values("mediana")
    .reset_index()
)

print("Dias até recompra por família (ordenado por mediana):")
print(stats_familia.to_string(index=False))

# ── Timing ideal de contato ──
# Recomendação: contatar no P25 — momento em que os clientes mais rápidos já recompram
# É a janela de oportunidade antes que o cliente esqueça
print(f"""
⏱️  TIMING IDEAL DE CONTATO POR FAMÍLIA
(recomendação: acionar no P25 — antes que a janela feche)
""")
for _, row in stats_familia.iterrows():
    print(f"   {row['familia_pedido']:<15} → contatar em ~{int(row['p25'])} dias "
          f"(mediana: {int(row['mediana'])}d | P75: {int(row['p75'])}d)")

# ── Visualização ──
top_familias = stats_familia["familia_pedido"].tolist()
dados_box = [
    df_recompra[df_recompra["familia_pedido"] == f]["dias_recompra"].values
    for f in top_familias
]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("white")
fig.suptitle("Dias até Recompra por Família de Produto — Dermage",
             fontsize=15, fontweight="bold", color=COR_BASE, y=1.01)

# ── Boxplot ──
ax = axes[0]
bp = ax.boxplot(
    dados_box,
    vert=False,
    patch_artist=True,
    medianprops=dict(color=COR_ROSA, linewidth=2.5),
    whiskerprops=dict(color="#AAAAAA"),
    capprops=dict(color="#AAAAAA"),
    flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#CCCCCC"),
)
cores_box = [COR_ROXO if i % 2 == 0 else COR_BASE
             for i in range(len(top_familias))]
for patch, cor in zip(bp["boxes"], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.6)

ax.set_yticks(range(1, len(top_familias) + 1))
ax.set_yticklabels(top_familias, fontsize=10)
ax.set_xlabel("Dias até recompra", fontsize=11)
ax.set_title("Distribuição de dias até recompra\n(linha rosa = mediana)", fontweight="bold")
ax.axvline(117, color="#AAAAAA", linestyle="--", linewidth=1,
           label="Mediana geral (117d)")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Tabela de timing ideal ──
ax2 = axes[1]
ax2.axis("off")

col_labels = ["Família", "Contatar em", "Mediana", "75% recompra"]
cell_data  = [
    [row["familia_pedido"],
     f"{int(row['p25'])} dias",
     f"{int(row['mediana'])} dias",
     f"{int(row['p75'])} dias"]
    for _, row in stats_familia.iterrows()
]

tabela = ax2.table(
    cellText=cell_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.8)

# Estilo do header
for j in range(len(col_labels)):
    tabela[0, j].set_facecolor(COR_BASE)
    tabela[0, j].set_text_props(color="white", fontweight="bold")

# Destaca coluna "Contatar em"
for i in range(1, len(cell_data) + 1):
    tabela[i, 0].set_facecolor("#F2F2F2")
    tabela[i, 1].set_facecolor("#FDE8EF")
    tabela[i, 1].set_text_props(color=COR_ROSA, fontweight="bold")

ax2.set_title("Timing ideal de contato por família\n(P25 = janela de oportunidade)",
              fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("recompra_por_familia.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\n✅ Salvo: recompra_por_familia.png")
plt.show()

"""
ANÁLISE POR SEGMENTO RFM — Dermage Datathon Girls in Tech 2026
Roda todas as análises para: Campeões, Em Risco, Hibernando
Depende do script principal já ter rodado (df_merge + rfm disponíveis)
"""
# ── Renomeia segmentos para apresentação ──
RENAME_SEGMENTOS = {
    "Campeões":   "Skin Glow",
    "Em Risco":   "Skin Bloom",
    "Hibernando": "Skin SOS",
}
rfm["segmento"] = rfm["segmento"].replace(RENAME_SEGMENTOS)

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
SEGMENTOS_ALVO = ["Skin Glow", "Skin Bloom", "Skin SOS"]
CORES_SEG = {
    "Skin Glow":  "#E75480",
    "Skin Bloom": "#F4A261",
    "Skin SOS":   "#6A5ACD",
}
CHURN_DIAS         = 180
TICKET_RECOMPRA    = 242.09
TAXA_RECOMPRA_REAL = 0.235
INCREMENTO         = 0.25

# ──────────────────────────────────────────
# 0. BASE POR SEGMENTO
# ──────────────────────────────────────────
print("=" * 60)
print("0. PREPARANDO BASE POR SEGMENTO RFM")
print("=" * 60)

seg_map = rfm.set_index("cli_document")["segmento"].to_dict()

df_base = (
    df_merge
    .dropna(subset=["cli_document", "data_pedido"])
    .drop_duplicates(subset=["cli_document", "order_id"])
    .copy()
)
df_base["segmento"] = df_base["cli_document"].map(seg_map)

familia_pedido = (
    df_merge.dropna(subset=["order_id", "familia"])
    .groupby("order_id")["familia"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"familia": "familia_pedido"})
)
df_base = df_base.merge(familia_pedido, on="order_id", how="left")

DATA_REF = df_base["data_pedido"].max()

for seg in SEGMENTOS_ALVO:
    n = (df_base["segmento"] == seg).sum()
    c = df_base[df_base["segmento"] == seg]["cli_document"].nunique()
    print(f"  {seg:<20} → {n:,} linhas | {c:,} clientes únicos")

# ──────────────────────────────────────────
# FUNÇÃO AUXILIAR
# ──────────────────────────────────────────
def analisar_segmento(seg, df_seg, cor):

    print(f"\n{'='*60}")
    print(f"SEGMENTO: {seg.upper()}")
    print(f"{'='*60}")

    df_s = df_seg.copy().sort_values(["cli_document", "data_pedido"])
    df_s["n_compra"] = df_s.groupby("cli_document").cumcount() + 1

    clientes_seg = df_s["cli_document"].nunique()
    ticket_med   = df_s.drop_duplicates("order_id")["valor_pedido"].mean()
    n_pedidos    = df_s["order_id"].nunique()

    # Taxa de recompra
    total_cli = df_s["cli_document"].nunique()
    cli_2x    = df_s[df_s["n_compra"] >= 2]["cli_document"].nunique()
    taxa_rec  = cli_2x / total_cli * 100 if total_cli > 0 else 0

    # Dias até 2ª compra
    primeira = df_s[df_s["n_compra"] == 1][["cli_document","data_pedido"]].rename(columns={"data_pedido":"data_1"})
    segunda  = df_s[df_s["n_compra"] == 2][["cli_document","data_pedido"]].rename(columns={"data_pedido":"data_2"})
    recompra = primeira.merge(segunda, on="cli_document")
    recompra["dias"] = (recompra["data_2"] - recompra["data_1"]).dt.days
    recompra = recompra[(recompra["dias"] > 0) & (recompra["dias"] <= 365)]

    mediana_dias = recompra["dias"].median() if len(recompra) > 0 else None
    p25_dias     = recompra["dias"].quantile(0.25) if len(recompra) > 0 else None

    # Churn
    clientes_ltv = df_s.groupby("cli_document").agg(
        ultima   = ("data_pedido", "max"),
        primeira = ("data_pedido", "min"),
    ).reset_index()
    clientes_ltv["dias_desde_ultima"]   = (DATA_REF - clientes_ltv["ultima"]).dt.days
    clientes_ltv["dias_desde_primeira"] = (DATA_REF - clientes_ltv["primeira"]).dt.days
    elegíveis  = clientes_ltv[clientes_ltv["dias_desde_primeira"] >= CHURN_DIAS]
    taxa_churn = (elegíveis["dias_desde_ultima"] >= CHURN_DIAS).mean() * 100 if len(elegíveis) > 0 else 0

    # LTV simples
    freq_anual = df_s.groupby("cli_document")["order_id"].count().mean() / \
                 max((df_s["data_pedido"].max() - df_s["data_pedido"].min()).days / 365, 0.1) * 12 \
                 if clientes_seg > 0 else 0
    taxa_churn_anual = min(taxa_churn / 100 * (12 / (CHURN_DIAS / 30)), 0.99)
    ltv = ticket_med * freq_anual * (1 / taxa_churn_anual) if taxa_churn_anual > 0 else 0

    # Top famílias
    fam_1a = df_s[df_s["n_compra"] == 1]["familia_pedido"].value_counts().head(5)
    fam_2a = df_s[df_s["n_compra"] == 2]["familia_pedido"].value_counts().head(5)

    # Timing por família
    df_s["data_proxima"]  = df_s.groupby("cli_document")["data_pedido"].shift(-1)
    df_s["dias_recompra"] = (df_s["data_proxima"] - df_s["data_pedido"]).dt.days
    df_rec = df_s.dropna(subset=["dias_recompra"]).copy()
    df_rec = df_rec[(df_rec["dias_recompra"] > 0) & (df_rec["dias_recompra"] <= 365)]

    stats_fam = (
        df_rec.groupby("familia_pedido")["dias_recompra"]
        .agg(obs="count", mediana="median",
             p25=lambda x: x.quantile(0.25),
             p75=lambda x: x.quantile(0.75))
        .round(1)
        .query("obs >= 30")
        .sort_values("mediana")
        .reset_index()
    )

    print(f"\n  Clientes:        {clientes_seg:,}")
    print(f"  Pedidos:         {n_pedidos:,}")
    print(f"  Ticket médio:    R$ {ticket_med:.2f}")
    print(f"  Taxa recompra:   {taxa_rec:.1f}%")
    print(f"  Mediana 2ª comp: {int(mediana_dias) if mediana_dias else 'N/A'} dias")
    print(f"  Timing contato:  {int(p25_dias) if p25_dias else 'N/A'} dias (P25)")
    print(f"  Churn:           {taxa_churn:.1f}%")
    print(f"  LTV estimado:    R$ {ltv:.2f}")
    print(f"\n  Top 3 famílias 1ª compra: {', '.join(fam_1a.head(3).index.tolist())}")
    print(f"  Top 3 famílias 2ª compra: {', '.join(fam_2a.head(3).index.tolist())}")

    return {
        "segmento":      seg,
        "cor":           cor,
        "clientes":      clientes_seg,
        "ticket_med":    round(ticket_med, 2),
        "taxa_rec":      round(taxa_rec, 1),
        "mediana_dias":  int(mediana_dias) if mediana_dias else 0,
        "p25_dias":      int(p25_dias) if p25_dias else 0,
        "taxa_churn":    round(taxa_churn, 1),
        "ltv":           round(ltv, 2),
        "fam_1a":        fam_1a,
        "fam_2a":        fam_2a,
        "stats_fam":     stats_fam,
        "recompra_dias": recompra,
    }

# ──────────────────────────────────────────
# RODA PARA OS 3 SEGMENTOS
# ──────────────────────────────────────────
resultados = {}
for seg in SEGMENTOS_ALVO:
    df_seg = df_base[df_base["segmento"] == seg].copy()
    if len(df_seg) == 0:
        print(f"\n⚠️  Segmento '{seg}' não encontrado na base.")
        continue
    resultados[seg] = analisar_segmento(seg, df_seg, CORES_SEG[seg])

# ──────────────────────────────────────────
# VISUALIZAÇÃO — 3 MÉTRICAS
# ──────────────────────────────────────────
segs  = list(resultados.keys())
cores = [resultados[s]["cor"] for s in segs]
x     = np.arange(len(segs))
width = 0.5

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("white")
fig.suptitle("Análise por Perfil de Cliente — Dermage",
             fontsize=15, fontweight="bold", color=COR_BASE, y=1.02)

# ── Ticket médio ──
ax = axes[0]
vals = [resultados[s]["ticket_med"] for s in segs]
bars = ax.bar(segs, vals, color=cores, edgecolor="white", width=width)
ax.set_title("Ticket Médio", fontweight="bold")
ax.set_ylabel("R$")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"R${v:.0f}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"R${v:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

# ── Taxa de recompra ──
ax = axes[1]
vals = [resultados[s]["taxa_rec"] for s in segs]
bars = ax.bar(segs, vals, color=cores, edgecolor="white", width=width)
ax.axhline(23.5, color="#AAAAAA", linestyle="--", linewidth=1.5,
           label="Média geral 23,5%")
ax.set_title("Taxa de Recompra", fontweight="bold")
ax.set_ylabel("%")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

# ── Mediana de dias até 2ª compra ──
ax = axes[2]
vals = [resultados[s]["mediana_dias"] for s in segs]
bars = ax.bar(segs, vals, color=cores, edgecolor="white", width=width)
ax.axhline(117, color="#AAAAAA", linestyle="--", linewidth=1.5,
           label="Mediana geral 117d")
ax.set_title("Mediana de Dias até 2ª Compra", fontweight="bold")
ax.set_ylabel("Dias")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{v}d", ha="center", va="bottom", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig("analise_por_segmento.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\n✅ Salvo: analise_por_segmento.png")
plt.show()

# ──────────────────────────────────────────
# RESUMO EXECUTIVO
# ──────────────────────────────────────────
print(f"""
{'='*60}
RESUMO COMPARATIVO — 3 PERFIS DE CLIENTE
{'='*60}
{'Métrica':<25} {'Skin Glow':>12} {'Skin Bloom':>12} {'Skin SOS':>12}
{'-'*60}""")

metricas = [
    ("Clientes",        "clientes",    "{:,}"),
    ("Ticket médio",    "ticket_med",  "R$ {:.0f}"),
    ("Taxa recompra",   "taxa_rec",    "{:.1f}%"),
    ("Timing contato",  "p25_dias",    "{} dias"),
    ("Mediana recomp.", "mediana_dias","{} dias"),
    ("Churn",           "taxa_churn",  "{:.1f}%"),
    ("LTV estimado",    "ltv",         "R$ {:.0f}"),
]

for label, key, fmt in metricas:
    vals = [fmt.format(resultados[s][key]) if s in resultados else "N/A"
            for s in SEGMENTOS_ALVO]
    print(f"  {label:<23} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

print(f"""
{'='*60}
💡 INSIGHTS CHAVE

  Skin Glow:  maior LTV e menor churn → programa VIP e fidelização
  Skin Bloom: compravam bem mas pararam → reativar antes de virar Skin SOS
  Skin SOS:   alto churn → campanha de win-back com oferta exclusiva
{'='*60}
""")



#################################################





############ MRR — CENÁRIO 50% RECOMPRA #####################################

TAXA_RECOMPRA_REAL = 0.235
TAXA_RECOMPRA_CEN  = 0.50        # dobrar a taxa absoluta
TICKET_RECOMPRA    = 256.00      # tm médio das compras 2ª a 5ª
FREQ_MEDIA_ANTIGO  = 1.77        # frequência média clientes antigos
JANELA_MESES       = round(117 / 30)  # 4 meses

df_mrr = (
    df_merge
    .dropna(subset=["cli_document", "data_pedido", "tipo_cliente"])
    .drop_duplicates(subset=["cli_document", "order_id"])
    [["cli_document", "order_id", "data_pedido", "valor_pedido", "tipo_cliente"]]
    .copy()
)
df_mrr["mes"] = df_mrr["data_pedido"].dt.to_period("M")

novos_por_mes = (
    df_mrr[df_mrr["tipo_cliente"] == "NOVO"]
    .groupby("mes")["cli_document"]
    .nunique()
    .reset_index()
    .rename(columns={"cli_document": "novos"})
)

mensal = (
    df_mrr.groupby(["mes", "tipo_cliente"])["valor_pedido"]
    .sum().unstack(fill_value=0).reset_index()
)
mensal["mes_str"]  = mensal["mes"].astype(str)
mensal["mrr_real"] = mensal.get("NOVO", 0) + mensal.get("ANTIGO", 0)
mensal = mensal.merge(novos_por_mes, on="mes", how="left")
mensal["novos"] = mensal["novos"].fillna(0)
mensal["novos_elegíveis"] = mensal["novos"].shift(JANELA_MESES, fill_value=0)

# Incremental = clientes adicionais × frequência média × ticket recompra
mensal["recompras_adicionais"] = (
    mensal["novos_elegíveis"] * (TAXA_RECOMPRA_CEN - TAXA_RECOMPRA_REAL)
)
mensal["receita_incremental"] = (
    mensal["recompras_adicionais"] * FREQ_MEDIA_ANTIGO * TICKET_RECOMPRA
).round(2)
mensal["mrr_cenario"] = mensal["mrr_real"] + mensal["receita_incremental"]

# ── Métricas ──
receita_real    = mensal["mrr_real"].sum()
receita_cenario = mensal["mrr_cenario"].sum()
ganho_total     = receita_cenario - receita_real
crescimento_pct = ganho_total / receita_real * 100

print(f"""
💳 CENÁRIO — DOBRAR TAXA DE RECOMPRA
─────────────────────────────────────────
   Taxa real:             {TAXA_RECOMPRA_REAL*100:.1f}%
   Taxa cenário:          {TAXA_RECOMPRA_CEN*100:.1f}%
   Ticket recompra:       R$ {TICKET_RECOMPRA:.2f}
   Frequência (antigos):  {FREQ_MEDIA_ANTIGO}x
─────────────────────────────────────────
   Receita real:          R$ {receita_real:,.0f}
   Receita cenário:       R$ {receita_cenario:,.0f}
   Ganho incremental:     R$ {ganho_total:,.0f}
   Crescimento:           {crescimento_pct:.1f}%
─────────────────────────────────────────
""")

# ── Plot ──
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("white")

x     = np.arange(len(mensal))
width = 0.6

ax.bar(x, mensal.get("ANTIGO", 0),
       width=width, color=COR_ROXO, label="Clientes Antigos (real)",
       edgecolor="white", zorder=3)
ax.bar(x, mensal.get("NOVO", 0),
       width=width, bottom=mensal.get("ANTIGO", 0),
       color=COR_ROSA, label="Clientes Novos (real)",
       edgecolor="white", zorder=3)
ax.bar(x, mensal["receita_incremental"],
       width=width, bottom=mensal["mrr_real"],
       color=COR_VERDE, label="Recompras adicionais (cenário 50%)",
       edgecolor="white", alpha=0.85, zorder=3)
ax.plot(x, mensal["mrr_cenario"], color=COR_BASE, linewidth=2.5,
        marker="o", markersize=5, label="Total cenário", zorder=5)

# Anotação do ganho
ax.annotate(
    f"+ R$ {ganho_total/1e6:.1f}M em recompras\n"
    f"({crescimento_pct:.0f}% de crescimento)\n"
    f"ticket R$ {TICKET_RECOMPRA:.0f} × {FREQ_MEDIA_ANTIGO}x por cliente",
    xy=(x[-3], mensal["mrr_cenario"].iloc[-3]),
    xytext=(x[-9], mensal["mrr_cenario"].max() * 0.78),
    fontsize=9, color=COR_VERDE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COR_VERDE, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor=COR_VERDE, alpha=0.9)
)

ax.set_xticks(x)
ax.set_xticklabels(mensal["mes_str"], rotation=45, ha="right", fontsize=9)
ax.set_title(
    "MRR Real vs Cenário — Taxa de Recompra de 23,5% para 50%\n"
    f"(ticket R$ {TICKET_RECOMPRA:.0f} | frequência {FREQ_MEDIA_ANTIGO}x | janela {JANELA_MESES} meses)",
    fontsize=13, fontweight="bold", color=COR_BASE, pad=14
)
ax.set_ylabel("Receita Mensal (R$)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"R${v/1000:.0f}k"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("mrr_cenario_50.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✅ Salvo: mrr_cenario_50.png")
plt.show()