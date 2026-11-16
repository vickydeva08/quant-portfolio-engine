import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import pdfplumber


# ==========================================
# 1. EXTRACT MULTI-DATE TRADES (PDF / EXCEL)
# ==========================================
def load_trade_data(file_path: str) -> pd.DataFrame:
  """Extracts trade records from an Excel (.xlsx/.xls) or PDF (.pdf) file.

  Standardizes columns to: Date, Stock, Type, Quantity, Price.
  """
  if file_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path)
  elif file_path.lower().endswith(".pdf"):
    rows = []
    with pdfplumber.open(file_path) as pdf:
      for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
          for row in table:
            if row and any(row):
              rows.append([cell.strip() if cell else "" for cell in row])
    if not rows:
      raise ValueError("No structured tables found in the PDF file.")
    df = pd.DataFrame(rows[1:], columns=rows[0])
  else:
    raise ValueError("Unsupported file format. Please provide .xlsx or .pdf")

  # Standardize column names dynamically
  df.columns = [col.strip().lower() for col in df.columns]
  col_map = {}
  for col in df.columns:
    if "date" in col:
      col_map[col] = "Date"
    elif "stock" in col or "symbol" in col or "scrip" in col:
      col_map[col] = "Stock"
    elif "type" in col or "action" in col or "buy" in col:
      col_map[col] = "Type"
    elif "qty" in col or "quantity" in col or "shares" in col:
      col_map[col] = "Quantity"
    elif "price" in col or "rate" in col:
      col_map[col] = "Price"

  df = df.rename(columns=col_map)
  df["Date"] = pd.to_datetime(df["Date"])
  df["Quantity"] = pd.to_numeric(
      df["Quantity"].astype(str).str.replace(",", ""), errors="coerce"
  )
  df["Price"] = pd.to_numeric(
      df["Price"].astype(str).str.replace("₹", "").str.replace(",", ""),
      errors="coerce",
  )
  df["Type"] = df["Type"].str.strip().str.capitalize()
  df["Stock"] = df["Stock"].str.strip().str.upper()

  return df.dropna(subset=["Date", "Stock", "Quantity", "Price"]).sort_values(
      "Date"
  )


# ==========================================
# 2. WEIGHTED AVERAGE COST & LEDGER ENGINE
# ==========================================
def calculate_stock_ledger(
    df: pd.DataFrame, current_market_prices: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Calculates cumulative shares, weighted average price, and unrealized P&L

  across multiple purchase dates.
  """
  if current_market_prices is None:
    current_market_prices = {}

  ledger_rows = []
  summary_rows = []

  for stock, group in df.groupby("Stock"):
    cum_qty = 0.0
    cum_cost = 0.0
    avg_price = 0.0

    for _, row in group.iterrows():
      qty = row["Quantity"]
      price = row["Price"]
      txn_type = row["Type"]
      trade_value = qty * price

      if txn_type == "Buy":
        cum_cost += trade_value
        cum_qty += qty
        avg_price = cum_cost / cum_qty if cum_qty > 0 else 0.0
      elif txn_type == "Sell":
        cum_qty -= qty
        cum_cost = cum_qty * avg_price if cum_qty > 0 else 0.0

      ledger_rows.append({
          "Date": row["Date"].strftime("%d-%b-%Y"),
          "Stock": stock,
          "Type": txn_type,
          "Quantity": qty,
          "Trade Price (₹)": round(price, 2),
          "Trade Value (₹)": round(trade_value, 2),
          "Cumulative Shares": cum_qty,
          "Weighted Avg Price (₹)": round(avg_price, 2),
          "Total Invested Cost (₹)": round(cum_cost, 2),
      })

    cmp = current_market_prices.get(stock, avg_price)
    current_val = cum_qty * cmp
    unrealized_pnl = current_val - cum_cost
    pnl_pct = (unrealized_pnl / cum_cost * 100) if cum_cost > 0 else 0.0

    summary_rows.append({
        "Stock": stock,
        "Net Shares Held": cum_qty,
        "Weighted Avg Cost (₹)": round(avg_price, 2),
        "Total Invested (₹)": round(cum_cost, 2),
        "Current Market Price (₹)": round(cmp, 2),
        "Current Value (₹)": round(current_val, 2),
        "Unrealized P&L (₹)": round(unrealized_pnl, 2),
        "Return (%)": round(pnl_pct, 2),
    })

  return pd.DataFrame(ledger_rows), pd.DataFrame(summary_rows)


# ==========================================
# 3. PORTFOLIO RISK & PERFORMANCE METRICS
# ==========================================
def calculate_portfolio_metrics(
    portfolio_values: pd.Series, risk_free_rate: float = 0.06
) -> dict:
  """Calculates CAGR, Annualized Sharpe Ratio, and Maximum Drawdown."""
  df = portfolio_values.sort_index()
  start_val, end_val = df.iloc[0], df.iloc[-1]
  years = (df.index[-1] - df.index[0]).days / 365.25
  cagr = (end_val / start_val) ** (1 / years) - 1

  daily_returns = df.pct_change().dropna()
  daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
  excess_returns = daily_returns - daily_rf
  sharpe = (
      np.sqrt(252) * excess_returns.mean() / daily_returns.std()
      if daily_returns.std() != 0
      else 0.0
  )

  rolling_max = df.cummax()
  max_drawdown = ((df - rolling_max) / rolling_max).min()

  return {
      "Initial Portfolio Value (₹)": round(start_val, 2),
      "Ending Portfolio Value (₹)": round(end_val, 2),
      "CAGR (%)": round(cagr * 100, 2),
      "Annualized Sharpe Ratio": round(sharpe, 2),
      "Maximum Drawdown (%)": round(max_drawdown * 100, 2),
  }


# ==========================================
# 4. DCF INTRINSIC VALUE & SCHEDULE
# ==========================================
def generate_dcf_schedule(
    initial_fcf: float,
    growth_rate: float,
    terminal_growth_rate: float,
    wacc: float,
    years: int,
    shares_outstanding: float,
) -> tuple[pd.DataFrame, dict]:
  """Calculates DCF Intrinsic Value and generates the year-by-year FCF schedule."""
  rows = []
  current_fcf = initial_fcf

  for t in range(1, years + 1):
    current_fcf *= 1 + growth_rate
    discount_factor = 1 / ((1 + wacc) ** t)
    pv_fcf = current_fcf * discount_factor
    rows.append({
        "Year": f"Year {t}",
        "Projected FCF (₹)": round(current_fcf, 2),
        "Discount Factor": round(discount_factor, 4),
        "Present Value of FCF (₹)": round(pv_fcf, 2),
    })

  dcf_df = pd.DataFrame(rows)
  sum_pv_fcf = dcf_df["Present Value of FCF (₹)"].sum()

  terminal_value = (current_fcf * (1 + terminal_growth_rate)) / (
      wacc - terminal_growth_rate
  )
  pv_terminal_value = terminal_value / ((1 + wacc) ** years)
  total_intrinsic_value = sum_pv_fcf + pv_terminal_value
  intrinsic_value_per_share = total_intrinsic_value / shares_outstanding

  summary = {
      "Initial Free Cash Flow (₹)": round(initial_fcf, 2),
      "Stage 1 Growth Rate (%)": round(growth_rate * 100, 2),
      "Terminal Growth Rate (%)": round(terminal_growth_rate * 100, 2),
      "WACC / Discount Rate (%)": round(wacc * 100, 2),
      "Sum of PV of Cash Flows (₹)": round(sum_pv_fcf, 2),
      "Terminal Value (₹)": round(terminal_value, 2),
      "PV of Terminal Value (₹)": round(pv_terminal_value, 2),
      "Total Intrinsic Value (₹)": round(total_intrinsic_value, 2),
      "Intrinsic Value Per Share (₹)": round(intrinsic_value_per_share, 2),
  }

  return dcf_df, summary


# ==========================================
# 5. SIP FUTURE VALUE CALCULATOR
# ==========================================
def calculate_sip_future_value(
    monthly_investment: float, annual_return: float, years: int
) -> dict:
  """Calculates SIP Future Value, Total Invested, and Wealth Gain."""
  monthly_rate = annual_return / 12
  months = years * 12
  fv = (
      monthly_investment
      * (((1 + monthly_rate) ** months - 1) / monthly_rate)
      * (1 + monthly_rate)
  )
  invested = monthly_investment * months

  return {
      "Monthly SIP Amount (₹)": round(monthly_investment, 2),
      "Duration (Years)": years,
      "Expected Annual Return (%)": round(annual_return * 100, 2),
      "Total Amount Invested (₹)": round(invested, 2),
      "Estimated Wealth Gain (₹)": round(fv - invested, 2),
      "Expected Future Value (₹)": round(fv, 2),
  }


# ==========================================
# 6. UNIFIED FORMATTED EXCEL EXPORTER
# ==========================================
def export_master_excel(
    filename: str,
    holdings_summary: pd.DataFrame,
    trade_ledger: pd.DataFrame,
    portfolio_metrics: dict,
    dcf_summary: dict,
    dcf_schedule: pd.DataFrame,
    sip_metrics: dict,
):
  with pd.ExcelWriter(filename, engine="openpyxl") as writer:
    # Sheet 1: Holdings Summary & Trade Ledger
    holdings_summary.to_excel(
        writer, sheet_name="Holdings & Ledger", startrow=2, index=False
    )
    ledger_start = len(holdings_summary) + 6
    trade_ledger.to_excel(
        writer,
        sheet_name="Holdings & Ledger",
        startrow=ledger_start,
        index=False,
    )

    # Sheet 2: Portfolio Risk (CAGR, Sharpe, Drawdown) & SIP
    port_df = pd.DataFrame(
        list(portfolio_metrics.items()), columns=["Metric", "Value"]
    )
    sip_df = pd.DataFrame(
        list(sip_metrics.items()), columns=["SIP Parameter", "Value"]
    )
    port_df.to_excel(
        writer, sheet_name="Risk & SIP Analysis", startrow=2, index=False
    )
    sip_start = len(port_df) + 6
    sip_df.to_excel(
        writer,
        sheet_name="Risk & SIP Analysis",
        startrow=sip_start,
        index=False,
    )

    # Sheet 3: DCF Intrinsic Value
    dcf_sum_df = pd.DataFrame(
        list(dcf_summary.items()), columns=["DCF Parameter / Output", "Value"]
    )
    dcf_sum_df.to_excel(
        writer, sheet_name="DCF Valuation", startrow=2, index=False
    )
    dcf_sched_start = len(dcf_sum_df) + 6
    dcf_schedule.to_excel(
        writer,
        sheet_name="DCF Valuation",
        startrow=dcf_sched_start,
        index=False,
    )

    # Styling setup
    header_fill = PatternFill(
        start_color="1F4E78", end_color="1F4E78", fill_type="solid"
    )
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=13, bold=True, color="1F4E78")
    zebra_fill = PatternFill(
        start_color="F2F5F9", end_color="F2F5F9", fill_type="solid"
    )
    thin_border = Side(border_style="thin", color="D9D9D9")
    border = Border(
        left=thin_border, right=thin_border, top=thin_border, bottom=thin_border
    )

    def format_block(ws, header_row, num_rows, num_cols, title_text, title_row):
      ws.cell(row=title_row, column=1, value=title_text).font = title_font
      for c in range(1, num_cols + 1):
        cell = ws.cell(row=header_row, column=c)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

      for r in range(header_row + 1, header_row + num_rows + 1):
        row_label = str(ws.cell(row=r, column=1).value)
        for c in range(1, num_cols + 1):
          cell = ws.cell(row=r, column=c)
          header_label = str(ws.cell(row=header_row, column=c).value)
          cell.border = border
          if (r - header_row) % 2 == 0:
            cell.fill = zebra_fill

          if isinstance(cell.value, (int, float)):
            cell.alignment = Alignment(horizontal="right", vertical="center")
            if "(₹)" in row_label or "(₹)" in header_label:
              cell.number_format = "₹#,##0.00"
            elif "(%)" in row_label or "(%)" in header_label:
              cell.number_format = '0.00"%"'
            elif "Discount Factor" in header_label:
              cell.number_format = "0.0000"
            elif "Quantity" in header_label or "Shares" in header_label:
              cell.number_format = "#,##0"
            else:
              cell.number_format = "#,##0.00"
          else:
            cell.alignment = Alignment(
                horizontal="left" if c == 1 else "center", vertical="center"
            )

    # Apply formatting to all 3 sheets
    ws1 = writer.sheets["Holdings & Ledger"]
    format_block(
        ws1,
        3,
        len(holdings_summary),
        len(holdings_summary.columns),
        "1. Portfolio Holdings & Weighted Average Cost Summary",
        2,
    )
    format_block(
        ws1,
        ledger_start + 1,
        len(trade_ledger),
        len(trade_ledger.columns),
        "2. Chronological Multi-Date Trade Ledger",
        ledger_start,
    )

    ws2 = writer.sheets["Risk & SIP Analysis"]
    format_block(
        ws2,
        3,
        len(port_df),
        2,
        "1. Portfolio Performance & Risk (CAGR, Sharpe Ratio, Max Drawdown)",
        2,
    )
    format_block(
        ws2,
        sip_start + 1,
        len(sip_df),
        2,
        "2. Systematic Investment Plan (SIP) Future Value Projection",
        sip_start,
    )

    ws3 = writer.sheets["DCF Valuation"]
    format_block(
        ws3,
        3,
        len(dcf_sum_df),
        2,
        "1. Discounted Cash Flow (DCF) Intrinsic Value Summary",
        2,
    )
    format_block(
        ws3,
        dcf_sched_start + 1,
        len(dcf_schedule),
        len(dcf_schedule.columns),
        "2. Year-by-Year Free Cash Flow Schedule",
        dcf_sched_start,
    )

    # Auto-fit column widths across all sheets
    title_rows = {2, ledger_start, sip_start, dcf_sched_start}
    for ws in [ws1, ws2, ws3]:
      ws.views.sheetView[0].showGridLines = True
      for col in ws.columns:
        max_len = max(
            (
                len(str(c.value))
                for c in col
                if c.value is not None and c.row not in title_rows
            ),
            default=14,
        )
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(
            max_len + 5, 18
        )

  print(f"Master workbook successfully exported: {filename}")


# ==========================================
# 7. EXECUTION EXAMPLE
# ==========================================
if __name__ == "__main__":
  # Step A: Load multi-date trades (replace with load_trade_data("file.pdf") or "file.xlsx")
  apc_trades = pd.DataFrame([
      {
          "Date": "2025-12-20",
          "Stock": "APC",
          "Type": "Buy",
          "Quantity": 100,
          "Price": 10.0,
      },
      {
          "Date": "2025-12-27",
          "Stock": "APC",
          "Type": "Buy",
          "Quantity": 100,
          "Price": 15.0,
      },
      {
          "Date": "2026-02-20",
          "Stock": "APC",
          "Type": "Buy",
          "Quantity": 200,
          "Price": 2.0,
      },
  ])
  apc_trades["Date"] = pd.to_datetime(apc_trades["Date"])
  ledger_df, summary_df = calculate_stock_ledger(
      apc_trades, current_market_prices={"APC": 8.50}
  )

  # Step B: Portfolio Risk Metrics (CAGR, Sharpe Ratio, Max Drawdown)
  dates = pd.date_range(start="2023-01-01", end="2026-01-01", freq="B")
  np.random.seed(42)
  sim_returns = np.random.normal(0.0004, 0.01, len(dates))
  portfolio_series = pd.Series(
      100000.0 * (1 + sim_returns).cumprod(), index=dates
  )
  risk_metrics = calculate_portfolio_metrics(
      portfolio_series, risk_free_rate=0.06
  )

  # Step C: DCF Intrinsic Value
  dcf_sched, dcf_sum = generate_dcf_schedule(
      initial_fcf=10000000.0,
      growth_rate=0.12,
      terminal_growth_rate=0.05,
      wacc=0.11,
      years=5,
      shares_outstanding=5000000,
  )

  # Step D: SIP Future Value
  sip_res = calculate_sip_future_value(
      monthly_investment=25000.0, annual_return=0.14, years=10
  )

  # Step E: Export all 3 sheets into one Excel file
  export_master_excel(
      filename="Complete_Portfolio_Analysis.xlsx",
      holdings_summary=summary_df,
      trade_ledger=ledger_df,
      portfolio_metrics=risk_metrics,
      dcf_summary=dcf_sum,
      dcf_schedule=dcf_sched,
      sip_metrics=sip_res,
  )