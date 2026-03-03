from __future__ import annotations

from rich.console import Console
from rich.text import Text

console = Console(highlight=False)

W = 56
RULE_CHAR = "─"
INDENT = "      "


def _rule(title: str | None = None) -> None:
    if title:
        label = f" {title} "
        pad = W - len(label)
        left = pad // 2
        right = pad - left
        line = RULE_CHAR * left + label + RULE_CHAR * right
    else:
        line = RULE_CHAR * W
    console.print(Text(line, style="dim"))


def _heading(number: str, title: str, meta: str | None = None) -> None:
    console.print()
    t = Text()
    t.append(number, style="dim")
    t.append("    ", style="")
    t.append(f'"{title}"', style="bold")
    if meta:
        t.append(f"    {meta}", style="dim")
    console.print(t)
    console.print()


def _kv(label: str, value: str, bold_value: bool = False) -> None:
    val_style = "bold" if bold_value else ""
    t = Text()
    t.append(f"{INDENT}{label:<22}", style="dim")
    t.append(value, style=val_style)
    console.print(t)


def header(title: str, subtitle: str | None = None) -> None:
    _rule(title.upper())
    if subtitle:
        console.print(Text(f"{INDENT}{subtitle}", style="dim"))


def entry(term: str, definition: str) -> None:
    _kv(term, definition)


def fold_header(fold: int, train_size: int, test_size: int) -> None:
    _heading(
        f"{fold:02d}",
        f"FOLD {fold}",
        f"{train_size:,} train / {test_size:,} test",
    )


def horizon_table(
    horizon_data: list[dict],
    ensemble_r2: float,
    ensemble_mse: float,
    n_horizons: int,
    n_samples: int,
) -> None:
    console.print(Text(f"{INDENT}{n_samples:,} samples", style="dim"))
    console.print()

    hdr = Text()
    hdr.append(f"{INDENT}{'horizon':<12}{'r²':<10}{'mse':<12}{'weight':<10}", style="dim")
    console.print(hdr)
    console.print(Text(f"{INDENT}{RULE_CHAR * 42}", style="dim"))

    for h in horizon_data:
        t = Text()
        t.append(f"{INDENT}{h['label']:<12}", style="dim")
        t.append(f"{h['r2']:<10.3f}", style="bold" if h["r2"] > 0.95 else "")
        t.append(f"{h['mse']:<12.4f}", style="dim")
        t.append(f"{h['weight']:<10.3f}", style="dim")
        console.print(t)

    console.print(Text(f"{INDENT}{RULE_CHAR * 42}", style="dim"))
    t = Text()
    t.append(f"{INDENT}{'ensemble':<12}", style="bold")
    t.append(f"{ensemble_r2:<10.3f}", style="bold")
    t.append(f"{ensemble_mse:<12.4f}", style="dim")
    t.append(f"{'1.000':<10}", style="dim")
    console.print(t)
    console.print()


def model_table(
    model_data: list[dict],
    ensemble_r2: float,
    ensemble_mse: float,
) -> None:
    hdr = Text()
    hdr.append(f"{INDENT}{'model':<16}{'r²':<10}{'mse':<12}{'weight':<10}", style="dim")
    console.print(hdr)
    console.print(Text(f"{INDENT}{RULE_CHAR * 46}", style="dim"))

    for m in model_data:
        t = Text()
        t.append(f"{INDENT}{m['name']:<16}", style="dim")
        t.append(f"{m['r2']:<10.3f}", style="bold" if m["r2"] > 0.95 else "")
        t.append(f"{m['mse']:<12.4f}", style="dim")
        t.append(f"{m['weight']:<10.3f}", style="dim")
        console.print(t)

    console.print(Text(f"{INDENT}{RULE_CHAR * 46}", style="dim"))
    t = Text()
    t.append(f"{INDENT}{'ensemble':<16}", style="bold")
    t.append(f"{ensemble_r2:<10.3f}", style="bold")
    t.append(f"{ensemble_mse:<12.4f}", style="dim")
    t.append(f"{'1.000':<10}", style="dim")
    console.print(t)
    console.print()


def fold_results(metrics: dict, metrics_no_exits: dict, currency: str, fit_metrics: dict | None = None) -> None:
    console.print()

    if fit_metrics:
        gap = fit_metrics["train_r2"] - fit_metrics["test_r2"]
        overfit_flag = "  !" if gap > 0.15 else ""
        _kv("train r²", f"{fit_metrics['train_r2']:>14.3f}")
        _kv("test r²", f"{fit_metrics['test_r2']:>14.3f}")
        _kv("r² gap", f"{gap:>14.3f}{overfit_flag}", bold_value=gap > 0.15)
        _kv("train mse", f"{fit_metrics['train_mse']:>14.2f}")
        _kv("test mse", f"{fit_metrics['test_mse']:>14.2f}")
        console.print()

    pnl = metrics["total_pnl"]
    pnl_ne = metrics_no_exits["total_pnl"]

    _kv("pnl", f"{currency} {pnl:>10.2f}", bold_value=True)
    _kv("pnl (no exits)", f"{currency} {pnl_ne:>10.2f}")
    _kv("hit rate", f"{metrics['hit_rate']:>13.1f}%", bold_value=True)
    _kv("trades", f"{metrics['total_trades']:>14}")
    _kv("sharpe", f"{metrics['sharpe_ratio']:>14.2f}", bold_value=True)
    _kv("sortino", f"{metrics['sortino_ratio']:>14.2f}")
    _kv("max drawdown", f"{currency} {metrics['max_drawdown']:>10.2f}")
    console.print()


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
    naive_pnl: float | None = None,
    naive_sharpe: float | None = None,
    naive_hit_rate: float | None = None,
    sharpe_ci: tuple[float, float] | None = None,
    sharpe_p: float | None = None,
) -> None:
    console.print()
    _rule('"BACKTEST"')
    console.print(Text(f"{INDENT}{n_folds} folds / expanding walk-forward", style="dim"))
    console.print()

    _kv("pnl", f"{currency} {avg_pnl:>10.2f}", bold_value=True)
    _kv("max drawdown", f"{currency} {avg_dd:>10.2f}")
    _kv("sharpe", f"{avg_sharpe:>14.2f}", bold_value=True)
    _kv("sortino", f"{avg_sortino:>14.2f}")
    _kv("hit rate", f"{avg_hit_rate:>13.1f}%", bold_value=True)
    _kv("consensus", f"{avg_consensus:>13.1%}")
    _kv("hours traded", f"{avg_traded:>13.1f}%")
    _kv("avg trades / fold", f"{avg_trades:>14.0f}")
    _kv("avg position size", f"{avg_position:>14.2f}")

    if sharpe_ci is not None and sharpe_p is not None:
        console.print()
        _kv("sharpe 95% ci", f"[{sharpe_ci[0]:>6.2f}, {sharpe_ci[1]:>5.2f}]")
        _kv("sharpe p-value", f"{sharpe_p:>14.3f}", bold_value=sharpe_p < 0.05)

    if naive_pnl is not None:
        console.print()
        _rule('"NAIVE BENCHMARK"')
        console.print(Text(f"{INDENT}trade every analyst signal, no meta-label gating", style="dim"))
        console.print()
        _kv("naive pnl", f"{currency} {naive_pnl:>10.2f}")
        _kv("naive sharpe", f"{naive_sharpe:>14.2f}" if naive_sharpe is not None else "           n/a")
        _kv("naive hit rate", f"{naive_hit_rate:>13.1f}%" if naive_hit_rate is not None else "           n/a")
        console.print()

        pnl_lift = avg_pnl - naive_pnl if naive_pnl else 0
        sharpe_lift = avg_sharpe - (naive_sharpe or 0)
        hit_lift = avg_hit_rate - (naive_hit_rate or 0)
        _kv("pnl lift", f"{currency} {pnl_lift:>10.2f}", bold_value=pnl_lift > 0)
        _kv("sharpe lift", f"{sharpe_lift:>+14.2f}", bold_value=sharpe_lift > 0)
        _kv("hit rate lift", f"{hit_lift:>+13.1f}pp", bold_value=hit_lift > 0)

    console.print()
    _rule()
    console.print()


def status(message: str, style: str = "dim") -> None:
    console.print(Text(f"      > {message}", style=style))


def data_loaded(filepath: str, n_rows: int, n_cols: int) -> None:
    _heading("00", "DATA", f"{n_rows:,} rows / {n_cols} features / {filepath}")


def warning(message: str) -> None:
    console.print(Text(f"      ! {message}", style="dim"))
