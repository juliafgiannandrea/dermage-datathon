"""
EDA - Dermage Datathon Girls in Tech 2026
Base de Produtos (base_produtos.csv)
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
ARQUIVO_PRODUTOS = "base_produtos.csv"
ARQUIVO_PEDIDOS  = "base_pedidos.csv"    # para cruzamento (opcional)

COR_1    = "#E75480"
COR_2    = "#6A5ACD"
COR_3    = "#2E8B57"
COR_BASE = "#2E4057"
PALETTE  = [COR_1, COR_2, COR_3, "#F4A261", "#264653", "#E9C46A", "#A8DADC"]

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
    ARQUIVO_PRODUTOS,
    sep=";",
    encoding="utf-8-sig",
    decimal=".",
    dtype=str,            # tudo string primeiro para inspecionar
)

print(f"\nShape bruto: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print(f"\nColunas: {list(df.columns)}")

# ── Renomeia para nomes pythônicos ──
rename = {
    "Origin":                "canal",
    "Order":                 "order_id",
    "Creation Date":         "data",
    "Client Document":       "cli_document",
    "UF":                    "uf",
    "Status":                "status",
    "Payment System Name":   "pagamento",
    "Installments":          "parcelas",
    "Payment Value":         "valor_pago",
    "Quantity_SKU":          "qtd_sku",
    "ID_SKU":                "id_sku",
    "Reference Code":        "ref_code",
    "SKU Name":              "sku_name",
    "SKU Value":             "sku_value",
    "SKU Selling Price":     "sku_preco_venda",
    "SKU Total Price":       "sku_total",
    "Total Value":           "total_pedido",
    "Discounts Totals":      "desconto",
    "Seller Name":           "seller",
}
df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

# ── Tipos numéricos ──
cols_num = ["valor_pago", "qtd_sku", "sku_value", "sku_preco_venda",
            "sku_total", "total_pedido", "desconto", "parcelas"]
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ── Data ──
if "data" in df.columns:
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

# ── Filtra apenas faturados ──
STATUS_VALIDO = ["Faturado", "invoiced", "Faturada"]
df_val = df[df["status"].isin(STATUS_VALIDO)].copy()
print(f"\nLinhas válidas (faturado): {len(df_val):,}")

# ──────────────────────────────────────────
# 2. EXTRAÇÃO DA FAMÍLIA DO PRODUTO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. FAMÍLIAS DE PRODUTO")
print("=" * 60)

# O nome da família é a primeira palavra(s) do SKU Name.
# Estratégia: pega o primeiro token antes de qualquer número ou vírgula.
# Ajuste conforme o familias.csv quando disponível.

def extrair_familia(nome):
    if pd.isna(nome):
        return "Desconhecido"
    # Remove encoding artefatos comuns
    nome = nome.strip()
    # Divide no primeiro espaço seguido de número ou em palavras-chave de variante
    import re
    # Tenta pegar as 1-2 primeiras palavras como família
    tokens = nome.split()
    # Famílias conhecidas da Dermage (adicione conforme familias.csv)
    familias_conhecidas = [
    "Photoage", "Secatriz", "Hyaluage", "Improve", "Revicare",
    "Compose", "Glycolique", "Revox", "Vinocare", "Clarité",
    "Revitrat", "Ineout", "EXOCARE", "Age Inverse",
]

    for fam in familias_conhecidas:
        if nome.lower().startswith(fam.lower()):
            return fam
    # Fallback: primeira palavra
    return tokens[0] if tokens else "Desconhecido"

df_val["familia"] = df_val["sku_name"].apply(extrair_familia)

print(f"\nFamílias identificadas: {df_val['familia'].nunique()}")
print(f"\nTop 15 famílias por linhas de produto:")
print(df_val["familia"].value_counts().head(15))

# ──────────────────────────────────────────
# 3. QUALIDADE DOS DADOS
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("3. QUALIDADE DOS DADOS")
print("=" * 60)

nulos = df_val.isnull().sum()
pct   = (nulos / len(df_val) * 100).round(2)
print(pd.DataFrame({"Nulos": nulos, "% Nulos": pct, "Únicos": df_val.nunique()}).to_string())

print(f"\nPeríodo: {df_val['data'].min().date()} → {df_val['data'].max().date()}")
print(f"Pedidos únicos: {df_val['order_id'].nunique():,}")
print(f"Clientes únicos: {df_val['cli_document'].nunique():,}")
print(f"SKUs únicos: {df_val['id_sku'].nunique():,}")

# ──────────────────────────────────────────
# 4. CANAL DE VENDA
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CANAL DE VENDA (Fulfillment vs Marketplace)")
print("=" * 60)

# ⚠️ ATENÇÃO: total_pedido repete por linha — somamos por pedido único
pedidos_unicos = df_val.drop_duplicates(subset="order_id")[["order_id","canal","total_pedido","data","uf"]].copy()

canal_summary = pedidos_unicos.groupby("canal").agg(
    pedidos       = ("order_id",      "count"),
    receita_total = ("total_pedido",  "sum"),
    ticket_medio  = ("total_pedido",  "mean"),
).round(2)
canal_summary["% pedidos"] = (canal_summary["pedidos"] / canal_summary["pedidos"].sum() * 100).round(1)
print(f"\n{canal_summary}")

# ──────────────────────────────────────────
# 5. ANÁLISE DE PRODUTO PORTA DE ENTRADA
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. PRODUTO PORTA DE ENTRADA (1ª vs 2ª compra)")
print("=" * 60)

# Ordena compras por cliente e data
df_ord = df_val.sort_values(["cli_document", "data"]).copy()
df_ord["n_compra"] = df_ord.groupby("cli_document")["order_id"].transform(
    lambda x: x.map({v: i+1 for i, v in enumerate(x.unique())})
)

# Top famílias na 1ª compra
primeira = df_ord[df_ord["n_compra"] == 1]
segunda  = df_ord[df_ord["n_compra"] == 2]

print("\nTop 10 famílias na 1ª compra:")
print(primeira["familia"].value_counts().head(10))

print("\nTop 10 famílias na 2ª compra:")
print(segunda["familia"].value_counts().head(10))

# Jornada: família na 1ª → família na 2ª
jornada = primeira[["cli_document","familia"]].rename(columns={"familia":"fam_1"}).merge(
    segunda[["cli_document","familia"]].rename(columns={"familia":"fam_2"}),
    on="cli_document"
)
print(f"\nTop 10 jornadas de produto (1ª → 2ª compra):")
print(jornada.groupby(["fam_1","fam_2"]).size().sort_values(ascending=False).head(10))

# ──────────────────────────────────────────
# 6. ANÁLISE REGIONAL
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ANÁLISE REGIONAL")
print("=" * 60)

uf_summary = pedidos_unicos.groupby("uf").agg(
    pedidos      = ("order_id",     "count"),
    receita      = ("total_pedido", "sum"),
    ticket_medio = ("total_pedido", "mean"),
).sort_values("pedidos", ascending=False).round(2)
uf_summary["% pedidos"] = (uf_summary["pedidos"] / uf_summary["pedidos"].sum() * 100).round(1)
print(f"\nTop 15 UFs:\n{uf_summary.head(15)}")

# ──────────────────────────────────────────
# 7. DESCONTO E PREÇO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("7. DESCONTO & PREÇO")
print("=" * 60)

df_val["pct_desconto"] = np.where(
    df_val["sku_value"] > 0,
    (1 - df_val["sku_preco_venda"] / df_val["sku_value"]) * 100,
    np.nan
)

print(f"\nDesconto médio por família (top 10):")
print(df_val.groupby("familia")["pct_desconto"].mean().sort_values(ascending=False).head(10).round(1))

print(f"\nEstatísticas de desconto (%):")
print(df_val["pct_desconto"].describe(percentiles=[.25,.5,.75,.90]).round(2))

# ──────────────────────────────────────────
# 8. MEIO DE PAGAMENTO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("8. MEIO DE PAGAMENTO")
print("=" * 60)

pag = pedidos_unicos.copy()
# Puxa pagamento do df original (pode ter múltiplos por pedido → pega o primeiro)
pag_info = df_val.groupby("order_id")["pagamento"].first().reset_index()
pag = pag.merge(pag_info, on="order_id", how="left")

print(f"\nDistribuição de pagamento:")
print(pag["pagamento"].value_counts().head(10))

print(f"\nParcelas médias por meio:")
parc_info = df_val.groupby("order_id")["parcelas"].first().reset_index()
pag = pag.merge(parc_info, on="order_id", how="left")
print(pag.groupby("pagamento")["parcelas"].mean().sort_values(ascending=False).head(10).round(2))

# ──────────────────────────────────────────
# 9. SAZONALIDADE POR FAMÍLIA
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("9. SAZONALIDADE POR FAMÍLIA")
print("=" * 60)

df_val["mes"] = df_val["data"].dt.to_period("M").astype(str)
top_fams = df_val["familia"].value_counts().head(5).index.tolist()

saz = df_val[df_val["familia"].isin(top_fams)].groupby(
    ["mes","familia"])["sku_total"].sum().unstack(fill_value=0)
print(f"\nReceita mensal por top 5 famílias (últimos 6 meses):\n{saz.tail(6)}")

# ──────────────────────────────────────────
# 10. VISUALIZAÇÕES
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("10. GERANDO VISUALIZAÇÕES...")
print("=" * 60)

fig = plt.figure(figsize=(20, 24))
fig.suptitle("EDA — Dermage | Base de Produtos", fontsize=18, fontweight="bold",
             color=COR_BASE, y=0.99)
gs = GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

# ── 10.1 Top famílias por receita ──
ax1 = fig.add_subplot(gs[0, 0])
top_fam_rec = df_val.groupby("familia")["sku_total"].sum().sort_values(ascending=True).tail(12)
colors_bar  = [COR_1 if i >= len(top_fam_rec)-3 else COR_2 for i in range(len(top_fam_rec))]
top_fam_rec.plot(kind="barh", ax=ax1, color=colors_bar, edgecolor="white")
ax1.set_title("Top Famílias por Receita Total", fontweight="bold")
ax1.set_xlabel("Receita (R$)")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"R${x/1e6:.1f}M" if x>=1e6 else f"R${x/1e3:.0f}k"))

# ── 10.2 Canal de venda ──
ax2 = fig.add_subplot(gs[0, 1])
canal_plot = canal_summary.reset_index()
bars = ax2.bar(canal_plot["canal"], canal_plot["pedidos"], color=[COR_1, COR_2], edgecolor="white", width=0.5)
ax2.set_title("Pedidos por Canal de Venda", fontweight="bold")
ax2.set_ylabel("Pedidos únicos")
for bar, row in zip(bars, canal_plot.itertuples()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{row.pedidos:,}\n({row._5:.1f}%)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

# ── 10.3 Ticket médio por canal ──
ax3 = fig.add_subplot(gs[1, 0])
bars2 = ax3.bar(canal_plot["canal"], canal_plot["ticket_medio"], color=[COR_1, COR_2], edgecolor="white", width=0.5)
ax3.set_title("Ticket Médio por Canal", fontweight="bold")
ax3.set_ylabel("R$")
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"R${bar.get_height():.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ── 10.4 Top 10 UFs ──
ax4 = fig.add_subplot(gs[1, 1])
top_uf = uf_summary.head(10).reset_index()
ax4.bar(top_uf["uf"], top_uf["pedidos"], color=COR_BASE, edgecolor="white", alpha=0.85)
ax4.set_title("Top 10 UFs por Volume de Pedidos", fontweight="bold")
ax4.set_ylabel("Pedidos")
ax4.tick_params(axis="x", rotation=0)
for p in ax4.patches:
    ax4.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=8)

# ── 10.5 1ª compra vs 2ª compra (família) ──
ax5 = fig.add_subplot(gs[2, 0])
fam_1 = primeira["familia"].value_counts().head(8)
fam_2 = segunda["familia"].value_counts().head(8)
all_fams = list(dict.fromkeys(list(fam_1.index) + list(fam_2.index)))[:8]
x = np.arange(len(all_fams))
w = 0.35
ax5.bar(x - w/2, [fam_1.get(f, 0) for f in all_fams], width=w, color=COR_1, label="1ª compra", edgecolor="white")
ax5.bar(x + w/2, [fam_2.get(f, 0) for f in all_fams], width=w, color=COR_2, label="2ª compra", edgecolor="white")
ax5.set_xticks(x)
ax5.set_xticklabels(all_fams, rotation=35, ha="right", fontsize=9)
ax5.set_title("Família na 1ª vs 2ª Compra", fontweight="bold")
ax5.set_ylabel("Ocorrências")
ax5.legend()

# ── 10.6 Distribuição de desconto ──
ax6 = fig.add_subplot(gs[2, 1])
desc_clip = df_val["pct_desconto"].clip(lower=0, upper=80).dropna()
sns.histplot(desc_clip, bins=40, ax=ax6, color=COR_2, edgecolor="white", alpha=0.8)
ax6.axvline(desc_clip.median(), color=COR_1, linewidth=2, linestyle="--",
            label=f"Mediana: {desc_clip.median():.1f}%")
ax6.set_title("Distribuição de Desconto por SKU (%)", fontweight="bold")
ax6.set_xlabel("Desconto (%)")
ax6.legend()

# ── 10.7 Sazonalidade top famílias ──
ax7 = fig.add_subplot(gs[3, :])
saz_plot = saz.copy()
for i, fam in enumerate(top_fams):
    if fam in saz_plot.columns:
        ax7.plot(saz_plot.index, saz_plot[fam], marker="o", linewidth=2,
                 markersize=4, color=PALETTE[i % len(PALETTE)], label=fam)
ax7.set_title("Receita Mensal — Top 5 Famílias", fontweight="bold")
ax7.set_ylabel("Receita (R$)")
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"R${v/1e3:.0f}k"))
ax7.tick_params(axis="x", rotation=45)
ax7.legend(title="Família", bbox_to_anchor=(1.01, 1), loc="upper left")
# mostra apenas alguns labels no eixo x para não poluir
ticks = ax7.get_xticks()
ax7.set_xticks(range(0, len(saz_plot), max(1, len(saz_plot)//12)))

plt.savefig("eda_dermage_produtos.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\n✅ Gráfico salvo: eda_dermage_produtos.png")
plt.show()

# ──────────────────────────────────────────
# 11. CRUZAMENTO COM BASE DE PEDIDOS (se disponível)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("11. CRUZAMENTO COM BASE DE PEDIDOS")
print("=" * 60)

try:
    df_ped = pd.read_csv(ARQUIVO_PEDIDOS, encoding="utf-8-sig", decimal=",", dtype=str,sep=';')
    df_ped["value"] = (
        df_ped["value"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    df_ped_val = df_ped[df_ped["status"] == "invoiced"].copy()

    # Clientes na base de produtos mas não na de pedidos (e vice-versa)
    cli_prod = set(df_val["cli_document"].dropna().unique())
    cli_ped  = set(df_ped_val["cli_document"].dropna().unique())
    print(f"\nClientes em produtos:    {len(cli_prod):,}")
    print(f"Clientes em pedidos:     {len(cli_ped):,}")
    print(f"Intersecção:             {len(cli_prod & cli_ped):,}")
    print(f"Apenas em produtos:      {len(cli_prod - cli_ped):,}")
    print(f"Apenas em pedidos:       {len(cli_ped - cli_prod):,}")

    # Enriquece base de produtos com tipo_cliente
    tipo_map = df_ped_val.set_index("cli_document")["tipo_cliente"].to_dict()
    df_val["tipo_cliente"] = df_val["cli_document"].map(tipo_map)

    print(f"\nReceita por tipo de cliente (base produtos):")
    print(df_val.groupby("tipo_cliente")["sku_total"].sum().apply(
        lambda x: f"R${x:,.2f}"))

    print("\nTop famílias por tipo de cliente:")
    print(df_val.groupby(["tipo_cliente","familia"])["sku_total"]
          .sum().groupby(level=0, group_keys=False)
          .nlargest(5).round(2))

except FileNotFoundError:
    print(f"\n⚠️  '{ARQUIVO_PEDIDOS}' não encontrado — pulando cruzamento.")

# ──────────────────────────────────────────
# 12. RESUMO EXECUTIVO
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("12. RESUMO EXECUTIVO — BASE DE PRODUTOS")
print("=" * 60)

top3_fam   = df_val["familia"].value_counts().head(3).index.tolist()
top_uf_str = uf_summary.head(5).index.tolist()
pct_mktpl  = canal_summary.loc["Marketplace","% pedidos"] if "Marketplace" in canal_summary.index else "N/A"
pct_full   = canal_summary.loc["Fulfillment","% pedidos"] if "Fulfillment" in canal_summary.index else "N/A"

print(f"""
🛒 BASE DE PRODUTOS
   • Total de linhas (SKUs faturados): {len(df_val):,}
   • Pedidos únicos: {df_val['order_id'].nunique():,}
   • Clientes únicos: {df_val['cli_document'].nunique():,}
   • SKUs distintos: {df_val['id_sku'].nunique():,}
   • Famílias identificadas: {df_val['familia'].nunique()}

📦 CANAL
   • Marketplace: {pct_mktpl}% dos pedidos
   • Fulfillment (e-com próprio): {pct_full}% dos pedidos

🗺️  REGIÕES TOP 5: {', '.join(top_uf_str)}

🏆 TOP 3 FAMÍLIAS: {', '.join(top3_fam)}

🏷️  DESCONTO
   • Desconto médio: {df_val['pct_desconto'].mean():.1f}%
   • Pedidos sem desconto: {(df_val['pct_desconto'] == 0).mean()*100:.1f}%
""")