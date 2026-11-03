"""Shared figure style: two-column layout, lowercase bold panel letters and
editable SVG text (Arial, falling back to Liberation Sans or DejaVu Sans)."""
import matplotlib

WIDTH = 8.8  # inches; every figure uses two panel columns at this width


def apply():
    matplotlib.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })


def size(rows):
    """Figure size for a two-column grid with the given number of rows."""
    return (WIDTH, 0.4 + 3.7 * rows)
