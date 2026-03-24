"""
EDA - Dermage Datathon Girls in Tech 2026
Base de Pedidos (base_pedidos.csv)
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
ARQUIVO = "base_pedidos.csv"   # ← ajuste o caminho se necessário
COR_NOVO   = "#E75480"
COR_ANTIGO = "#6A5ACD"
COR_BASE   = "#2E4057"
PALETTE    = [COR_NOVO, COR_ANTIGO]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})

# ──────────────────────────────────────────
# 1. CARREGAMENTO E LIMPEZA
# ──────────────────────────────────────────
print("=" * 60)
print("1. CARREGAMENTO DOS DADOS")
print("=" * 60)

df = pd.read_csv(
    ARQUIVO,
    sep=";",                  # ajuste para ";" se necessário
    encoding="utf-8-sig",     # lida com BOM (»¿)
    decimal=",",              # vírgula como separador decimal
    dtype={"value": str},     # lê como string primeiro para tratar vírgula
)

# Trata coluna de valor
df["value"] = (
    df["value"]
    .astype(str)
    .str.replace(".", "", regex=False)   # remove milhar
    .str.replace(",", ".", regex=False)  # vírgula → ponto
    .astype(float)
)

# Trata datas
for col in ["data", "data_tratada"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

# Remove colunas duplicadas se existirem
df = df.loc[:, ~df.columns.duplicated()]

print(f"\nShape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print(f"\nColunas: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nPrimeiras linhas:\n{df.head()}")

# ──────────────────────────────────────────
# 2. QUALIDADE DOS DADOS
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. QUALIDADE DOS DADOS")
print("=" * 60)

nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
qualidade = pd.DataFrame({"Nulos": nulos, "% Nulos": pct_nulos, "Únicos": df.nunique()})
print(f"\n{qualidade}")

print(f"\nPeríodo dos dados: {df['data'].min().date()} → {df['data'].max().date()}")
print(f"Meses cobertos: {df['data'].dt.to_period('M').nunique()}")

# Distribuição de status
print(f"\nDistribuição de status:")
print(df["status"].value_counts(dropna=False))

# Filtro: apenas pedidos válidos (invoiced)
df_valido = df[df["status"] == "invoiced"].copy()
print(f"\nPedidos invoiced: {len(df_valido):,} ({len(df_valido)/len(df)*100:.1f}%)")

# ──────────────────────────────────────────
# 3. ESTATÍSTICAS DESCRITIVAS - VALOR
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ESTATÍSTICAS DE VALOR (pedidos invoiced)")
print("=" * 60)

desc = df_valido["value"].describe(percentiles=[.25, .5, .75, .90, .95])
print(f"\n{desc.round(2)}")

# Detecta outliers via IQR
Q1, Q3 = df_valido["value"].quantile([.25, .75])
IQR = Q3 - Q1
outliers = df_valido[df_valido["value"] > Q3 + 3 * IQR]
print(f"\nOutliers (> Q3+3×IQR = R${Q3+3*IQR:.0f}): {len(outliers):,} pedidos")

# ──────────────────────────────────────────
# 4. PERFIL NOVO vs ANTIGO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("4. PERFIL DE CLIENTES: NOVO vs ANTIGO")
print("=" * 60)

perfil = df_valido.groupby("tipo_cliente").agg(
    pedidos    = ("orderid",      "count"),
    clientes   = ("cli_document", "nunique"),
    ticket_med = ("value",        "mean"),
    ticket_med_= ("value",        "median"),
    total_rev  = ("value",        "sum"),
).round(2)

perfil.columns = ["Pedidos", "Clientes únicos", "Ticket médio", "Ticket mediano", "Receita total"]
print(f"\n{perfil}")

# Pedidos por cliente
df_valido_grp = df_valido.groupby(["cli_document", "tipo_cliente"])["orderid"].count().reset_index()
df_valido_grp.columns = ["cli_document", "tipo_cliente", "n_pedidos"]
print(f"\nPedidos por cliente:")
print(df_valido_grp.groupby("tipo_cliente")["n_pedidos"].describe().round(2))

# ──────────────────────────────────────────
# 5. ANÁLISE DE RECOMPRA
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ANÁLISE DE RECOMPRA")
print("=" * 60)

# Ordena por cliente e data
df_ord = df_valido.sort_values(["cli_document", "data"]).copy()
df_ord["n_compra"] = df_ord.groupby("cli_document").cumcount() + 1

# Taxa de recompra geral
total_cli    = df_valido["cli_document"].nunique()
cli_2x       = df_ord[df_ord["n_compra"] >= 2]["cli_document"].nunique()
cli_3x       = df_ord[df_ord["n_compra"] >= 3]["cli_document"].nunique()
taxa_recompra = cli_2x / total_cli * 100

print(f"\nTotal de clientes únicos (invoiced): {total_cli:,}")
print(f"Clientes com ≥ 2 compras:  {cli_2x:,} ({taxa_recompra:.1f}%)")
print(f"Clientes com ≥ 3 compras:  {cli_3x:,} ({cli_3x/total_cli*100:.1f}%)")

# Tempo até 2ª compra
primeira  = df_ord[df_ord["n_compra"] == 1][["cli_document", "data"]].rename(columns={"data": "data_1"})
segunda   = df_ord[df_ord["n_compra"] == 2][["cli_document", "data"]].rename(columns={"data": "data_2"})
recompra  = primeira.merge(segunda, on="cli_document")
recompra["dias_ate_2a"] = (recompra["data_2"] - recompra["data_1"]).dt.days

print(f"\nDias até 2ª compra:")
print(recompra["dias_ate_2a"].describe(percentiles=[.25,.5,.75,.90]).round(1))

recompra["dias_ate_2a"].describe()

# Período crítico: buckets de tempo
bins   = [0, 30, 60, 90, 180, 365, 9999]
labels = ["0-30d", "31-60d", "61-90d", "91-180d", "181-365d", ">365d"]
recompra["bucket_dias"] = pd.cut(recompra["dias_ate_2a"], bins=bins, labels=labels)
print(f"\nDistribuição do tempo até 2ª compra:")
print(recompra["bucket_dias"].value_counts().sort_index())

# ──────────────────────────────────────────
# 6. SAZONALIDADE
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SAZONALIDADE")
print("=" * 60)

df_valido["mes"]       = df_valido["data"].dt.to_period("M")
df_valido["mes_nome"]  = df_valido["data"].dt.strftime("%Y-%m")

# Primeiras compras por mês
primeira_compra_cli = df_ord[df_ord["n_compra"] == 1].copy()
primeira_compra_cli["mes"] = primeira_compra_cli["data"].dt.to_period("M")

mensal = df_valido.groupby("mes_nome").agg(
    pedidos=("orderid","count"),
    receita=("value","sum"),
    novos_clientes=("cli_document","nunique"),
).reset_index()

print(f"\nResumo mensal (últimos 6 meses):\n{mensal.tail(6).to_string(index=False)}")

# ──────────────────────────────────────────
# 7. VISUALIZAÇÕES
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("7. GERANDO VISUALIZAÇÕES...")
print("=" * 60)

fig = plt.figure(figsize=(18, 20))
fig.suptitle("EDA — Dermage | Base de Pedidos", fontsize=18, fontweight="bold",
             color=COR_BASE, y=0.98)
gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 7.1 Distribuição de status ──
ax1 = fig.add_subplot(gs[0, 0])
status_counts = df["status"].value_counts()
cores_status = [COR_NOVO if s == "invoiced" else "#CCCCCC" for s in status_counts.index]
status_counts.plot(kind="bar", ax=ax1, color=cores_status, edgecolor="white")
ax1.set_title("Distribuição de Status dos Pedidos", fontweight="bold")
ax1.set_xlabel("")
ax1.set_ylabel("Quantidade")
ax1.tick_params(axis="x", rotation=30)
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=9)

# ── 7.2 Ticket médio novo vs antigo ──
ax2 = fig.add_subplot(gs[0, 1])
ticket_tipo = df_valido.groupby("tipo_cliente")["value"].mean()
bars = ax2.bar(ticket_tipo.index, ticket_tipo.values, color=PALETTE, edgecolor="white", width=0.5)
ax2.set_title("Ticket Médio: NOVO vs ANTIGO", fontweight="bold")
ax2.set_ylabel("R$")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"R${bar.get_height():.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ── 7.3 Distribuição de valor (sem outliers) ──
ax3 = fig.add_subplot(gs[1, 0])
upper = df_valido["value"].quantile(0.95)
sns.histplot(df_valido[df_valido["value"] <= upper]["value"], bins=40,
             ax=ax3, color=COR_ANTIGO, edgecolor="white", alpha=0.8)
ax3.axvline(df_valido["value"].median(), color=COR_NOVO, linewidth=2, linestyle="--",
            label=f"Mediana: R${df_valido['value'].median():.0f}")
ax3.set_title("Distribuição de Ticket (p95)", fontweight="bold")
ax3.set_xlabel("Valor do Pedido (R$)")
ax3.set_ylabel("Frequência")
ax3.legend()

# ── 7.4 Pedidos por mês ──
ax4 = fig.add_subplot(gs[1, 1])
mensal_plot = mensal.copy()
mensal_plot["mes_label"] = pd.to_datetime(mensal_plot["mes_nome"]).dt.strftime("%b/%y")
ax4.plot(mensal_plot["mes_label"], mensal_plot["pedidos"], marker="o",
         color=COR_BASE, linewidth=2, markersize=5)
ax4.fill_between(mensal_plot["mes_label"], mensal_plot["pedidos"], alpha=0.15, color=COR_BASE)
ax4.set_title("Volume de Pedidos por Mês", fontweight="bold")
ax4.set_ylabel("Pedidos (invoiced)")
ax4.tick_params(axis="x", rotation=45)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# ── 7.5 Tempo até 2ª compra ──
ax5 = fig.add_subplot(gs[2, 0])
bucket_counts = recompra["bucket_dias"].value_counts().sort_index()
colors_bucket = [COR_NOVO if i == 0 else COR_ANTIGO if i == 1 else "#AAAAAA"
                 for i in range(len(bucket_counts))]
bucket_counts.plot(kind="bar", ax=ax5, color=colors_bucket, edgecolor="white")
ax5.set_title("Tempo até 2ª Compra (Período Crítico)", fontweight="bold")
ax5.set_xlabel("Janela de Tempo")
ax5.set_ylabel("Clientes")
ax5.tick_params(axis="x", rotation=30)
for p in ax5.patches:
    ax5.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=9)

# ── 7.6 Nº de compras por cliente (distribuição) ──
ax6 = fig.add_subplot(gs[2, 1])
compras_por_cli = df_ord.groupby("cli_document")["n_compra"].max().value_counts().sort_index().head(10)
compras_por_cli.plot(kind="bar", ax=ax6, color=COR_ANTIGO, edgecolor="white")
ax6.set_title("Distribuição: Qtd de Compras por Cliente", fontweight="bold")
ax6.set_xlabel("Número de compras")
ax6.set_ylabel("Clientes")
ax6.tick_params(axis="x", rotation=0)
for p in ax6.patches:
    ax6.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=9)

# ── 7.7 Receita mensal por tipo de cliente ──
ax7 = fig.add_subplot(gs[3, :])
mensal_tipo = df_valido.groupby(["mes_nome", "tipo_cliente"])["value"].sum().unstack(fill_value=0).reset_index()
mensal_tipo["mes_label"] = pd.to_datetime(mensal_tipo["mes_nome"]).dt.strftime("%b/%y")
x = range(len(mensal_tipo))
width = 0.4
for i, (col, cor) in enumerate(zip(["NOVO", "ANTIGO"], PALETTE)):
    if col in mensal_tipo.columns:
        offset = (i - 0.5) * width
        ax7.bar([xi + offset for xi in x], mensal_tipo[col], width=width,
                color=cor, label=col, edgecolor="white", alpha=0.9)
ax7.set_xticks(list(x))
ax7.set_xticklabels(mensal_tipo["mes_label"], rotation=45)
ax7.set_title("Receita Mensal: Clientes NOVOS vs ANTIGOS", fontweight="bold")
ax7.set_ylabel("Receita (R$)")
ax7.legend(title="Tipo Cliente")
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"R${v/1000:.0f}k"))

plt.savefig("eda_dermage_pedidos.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("\n✅ Gráfico salvo: eda_dermage_pedidos.png")
plt.show()

# ──────────────────────────────────────────
# 8. RESUMO EXECUTIVO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("8. RESUMO EXECUTIVO — INSIGHTS PRELIMINARES")
print("=" * 60)

print(f"""
📦 BASE DE PEDIDOS
   • Total de pedidos: {len(df):,}
   • Pedidos válidos (invoiced): {len(df_valido):,}
   • Período: {df['data'].min().date()} → {df['data'].max().date()}

👥 CLIENTES
   • Total de clientes únicos: {total_cli:,}
   • Taxa de recompra (≥2 compras): {taxa_recompra:.1f}%
   • Clientes com ≥3 compras: {cli_3x/total_cli*100:.1f}%

⏱️  PERÍODO CRÍTICO
   • Mediana de dias até 2ª compra: {recompra['dias_ate_2a'].median():.0f} dias
   • 75% das recompras ocorrem em até {recompra['dias_ate_2a'].quantile(.75):.0f} dias
   
💰 TICKET
   • Ticket médio geral: R${df_valido['value'].mean():.2f}
   • Ticket mediano: R${df_valido['value'].median():.2f}
   • Ticket médio NOVO:   R${df_valido[df_valido['tipo_cliente']=='NOVO']['value'].mean():.2f}
   • Ticket médio ANTIGO: R${df_valido[df_valido['tipo_cliente']=='ANTIGO']['value'].mean():.2f}
""")