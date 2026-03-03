from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table, box
from rich.text import Text

console = Console()

ACCENT = "bright_cyan"
DIM = "dim"
BOLD = "bold"
METRIC_STYLE = "bold white"
HEADER_STYLE = "bold"


def header(title: str, subtitle: str | None = None) -> None:
    t = Text(title, style=HEADER_STYLE)
    if subtitle:
        t.append(f"\n{subtitle}", style=DIM)
    console.print()
    console.print(Panel(t, border_style=DIM, padding=(0, 2), expand=False))


def entry(term: str, definition: str) -> None:
    t = Text()
    t.append(term, style=BOLD)
    t.append(f"  {definition}", style=DIM)
    console.print(t)


def fold_header(fold: int, train_size: int, test_size: int) -> None:
    t = Text()
    t.append(f"Fold {fold}", style=HEADER_STYLE)
    t.append(f"   {train_size:,} train · {test_size:,} test samples", style=DIM)
    console.print()
    console.print(Panel(t, border_style=DIM, padding=(0, 2), expand=False))


def horizon_table(
    horizon_data: list[dict],
    ensemble_r2: float,
    ensemble_mse: float,
    n_horizons: int,
    n_samples: int,
) -> None:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        padding=(0, 2),
        title=f"[{DIM}]{n_samples:,} samples[/]",
        title_style=DIM,
    )
    table.add_column("horizon", style=DIM)
    table.add_column("r²", justify="right")
    table.add_column("mse", justify="right", style=DIM)
    table.add_column("weight", justify="right", style=DIM)

    for h in horizon_data:
        r2_text = f"[{ACCENT}]{h['r2']:.3f}[/]" if h["r2"] > 0.95 else f"{h['r2']:.3f}"
        table.add_row(
            h["label"],
            r2_text,
            f"{h['mse']:.4f}",
            f"{h['weight']:.3f}",
        )

    table.add_section()
    table.add_row(
        f"[{BOLD}]ensemble[/]",
        f"[{METRIC_STYLE}]{ensemble_r2:.3f}[/]",
        f"{ensemble_mse:.4f}",
        "1.000",
    )

    console.print(table)


def model_table(
    model_data: list[dict],
    ensemble_r2: float,
    ensemble_mse: float,
) -> None:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        padding=(0, 2),
    )
    table.add_column("model", style=DIM)
    table.add_column("r²", justify="right")
    table.add_column("mse", justify="right", style=DIM)
    table.add_column("weight", justify="right", style=DIM)

    for m in model_data:
        r2_text = f"[{ACCENT}]{m['r2']:.3f}[/]" if m["r2"] > 0.95 else f"{m['r2']:.3f}"
        table.add_row(
            m["name"],
            r2_text,
            f"{m['mse']:.4f}",
            f"{m['weight']:.3f}",
        )

    table.add_section()
    table.add_row(
        f"[{BOLD}]ensemble[/]",
        f"[{METRIC_STYLE}]{ensemble_r2:.3f}[/]",
        f"{ensemble_mse:.4f}",
        "1.000",
    )

    console.print(table)


def fold_results(metrics: dict, metrics_no_exits: dict, currency: str) -> None:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        padding=(0, 2),
    )
    table.add_column("", style=DIM)
    table.add_column("", justify="right")

    pnl = metrics["total_pnl"]
    pnl_style = "green" if pnl > 0 else "red"
    pnl_no_exit = metrics_no_exits["total_pnl"]
    pnl_ne_style = "green" if pnl_no_exit > 0 else "red"

    table.add_row("pnl (exits)", f"[{pnl_style}]{currency} {pnl:>8.2f}[/]")
    table.add_row("pnl (no exits)", f"[{pnl_ne_style}]{currency} {pnl_no_exit:>8.2f}[/]")
    table.add_row("hit rate", f"[{METRIC_STYLE}]{metrics['hit_rate']:>7.1f}%[/]")
    table.add_row("trades", f"{metrics['total_trades']:>8}")
    table.add_row("sharpe", f"[{ACCENT}]{metrics['sharpe_ratio']:>8.2f}[/]")
    table.add_row("sortino", f"{metrics['sortino_ratio']:>8.2f}")
    table.add_row("drawdown", f"[red]{currency} {metrics['max_drawdown']:>8.2f}[/]")

    console.print(table)


def backtest_summary(
    avg_pnl: float,
    avg_dd: float,
    avg_sharpe: float,
    avg_sortino: float,
    avg_hit_rate: float,
    avg_consensus: float,
    avg_traded: float,
    avg_trades: float,
    avg_position: float,
    currency: str,
    n_folds: int,
) -> None:
    console.print()

    title = Text()
    title.append("Backtest", style=HEADER_STYLE)
    title.append(f"   {n_folds} expanding walk-forward folds", style=DIM)

    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        padding=(0, 2),
    )
    table.add_column("", style=DIM)
    table.add_column("", justify="right")

    pnl_style = "green" if avg_pnl > 0 else "red"

    table.add_row("pnl", f"[{pnl_style}]{currency} {avg_pnl:.2f}[/]")
    table.add_row("max drawdown", f"[red]{currency} {avg_dd:.2f}[/]")
    table.add_row("sharpe", f"[{ACCENT}]{avg_sharpe:.2f}[/]")
    table.add_row("sortino", f"{avg_sortino:.2f}")
    table.add_row("hit rate", f"[{METRIC_STYLE}]{avg_hit_rate:.1f}%[/]")
    table.add_row("consensus", f"{avg_consensus:.1%}")
    table.add_row("hours traded", f"{avg_traded:.1f}%")
    table.add_row("avg trades / fold", f"{avg_trades:.0f}")
    table.add_row("avg position size", f"{avg_position:.2f}")

    console.print(Panel(table, title=title, title_align="left", border_style=DIM, padding=(0, 1)))
    console.print()


def status(message: str, style: str = DIM) -> None:
    console.print(f"  [{style}]{message}[/]")


def data_loaded(filepath: str, n_rows: int, n_cols: int) -> None:
    t = Text()
    t.append("Data", style=HEADER_STYLE)
    t.append(f"   {n_rows:,} rows · {n_cols} features", style=DIM)
    t.append(f"\n  {filepath}", style=DIM)
    console.print(Panel(t, border_style=DIM, padding=(0, 2), expand=False))


def warning(message: str) -> None:
    console.print(f"  [yellow dim]⚠ {message}[/]")
