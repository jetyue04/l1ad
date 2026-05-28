"""Generate a PDF report from a VICReg-WNAE training output directory."""

import re
import csv
import argparse
from datetime import datetime
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
    HRFlowable,
)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a2f4b")
TEAL   = colors.HexColor("#0d9488")
GOLD   = colors.HexColor("#d97706")
LIGHT  = colors.HexColor("#f0f4f8")
MID    = colors.HexColor("#cbd5e1")
WHITE  = colors.white
BLACK  = colors.black


# ── Page template with header / footer ───────────────────────────────────────
def _make_page_template(doc, run_label: str):
    def _header_footer(canvas, doc):
        canvas.saveState()
        w, h = letter

        # Top rule + title
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 0.55 * inch, w, 0.55 * inch, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(0.75 * inch, h - 0.35 * inch, "L1AD  ·  VICReg-WNAE Training Report")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 0.75 * inch, h - 0.35 * inch, run_label)

        # Bottom rule + page number
        canvas.setFillColor(MID)
        canvas.rect(0, 0, w, 0.4 * inch, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(w / 2, 0.15 * inch, f"Page {doc.page}")
        canvas.restoreState()

    pw, ph = letter
    frame = Frame(
        0.75 * inch, 0.55 * inch,
        pw - 1.5 * inch, ph - 1.1 * inch,
        id="main",
    )
    return PageTemplate(id="main", frames=[frame], onPage=_header_footer)


# ── Styles ────────────────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        base.add(ParagraphStyle(name=name, **kw))

    add("RunTitle",   parent=base["Title"],   fontSize=22, textColor=NAVY,
        spaceAfter=6,  spaceBefore=0,  leading=28, alignment=1)
    add("SubTitle",   parent=base["Normal"],  fontSize=11, textColor=TEAL,
        spaceAfter=4,  alignment=1)
    add("SectionH",  parent=base["Heading1"], fontSize=13, textColor=WHITE,
        spaceAfter=6,  spaceBefore=14, leading=18)
    add("EpochH",    parent=base["Heading2"], fontSize=11, textColor=NAVY,
        spaceAfter=4,  spaceBefore=10, leading=15)
    add("BestEpochH",parent=base["Heading2"], fontSize=11, textColor=GOLD,
        spaceAfter=4,  spaceBefore=10, leading=15)
    add("Mono",       parent=base["Normal"],  fontName="Courier", fontSize=8,
        leading=11)
    add("Caption",    parent=base["Normal"],  fontSize=7.5, textColor=colors.HexColor("#64748b"),
        alignment=1)
    return base


def section_heading(text, styles) -> list:
    """Navy banner heading."""
    label = Paragraph(f"<font color='white'><b>  {text}</b></font>", styles["SectionH"])
    tbl = Table([[label]], colWidths=[6.5 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [tbl, Spacer(1, 6)]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_best_epoch(info_path: Path):
    if not info_path.exists():
        return None
    m = re.search(r"Best\s*epoch:\s*(\d+)", info_path.read_text())
    return int(m.group(1)) if m else None


def _read_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _epoch_plot(feature_dir: Path, epoch: int):
    p = feature_dir / f"epoch_{epoch}.png"
    return p if p.exists() else None


def _add_config(cfg: dict, elements: list, mono, indent: int = 0):
    for k, v in cfg.items():
        pad = "&nbsp;" * indent
        if isinstance(v, dict):
            elements.append(Paragraph(f"{pad}<b>{k}:</b>", mono))
            _add_config(v, elements, mono, indent + 4)
        else:
            elements.append(Paragraph(f"{pad}{k}: <i>{v}</i>", mono))


# ── Main report builder ───────────────────────────────────────────────────────
def create_report(output_dir: Path, config=None, interval: int = 10):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise ValueError(f"Directory not found: {output_dir}")

    plots_dir          = output_dir / "sample_feature_1D_hist"
    training_loss_path = output_dir / "train_history.png"
    info_path          = output_dir / "info.txt"
    csv_path           = output_dir / "training.csv"
    pdf_path           = output_dir / "report.pdf"

    run_label  = output_dir.name
    best_epoch = _read_best_epoch(info_path)
    rows       = _read_csv(csv_path)

    # Feature directories
    feature_dirs = sorted(
        [d for d in plots_dir.iterdir() if d.is_dir() and "feature" in d.name],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    ) if plots_dir.exists() else []

    all_epochs = sorted(
        int(f.stem.replace("epoch_", ""))
        for f in (feature_dirs[0].glob("epoch_*.png") if feature_dirs else [])
    )
    selected = {e for e in all_epochs if e % interval == 0} | ({max(all_epochs)} if all_epochs else set())
    if best_epoch is not None:
        epoch_order = [best_epoch] + sorted(selected - {best_epoch})
    else:
        epoch_order = sorted(selected)

    # ── Document setup ────────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.55 * inch,
    )
    doc.addPageTemplates([_make_page_template(doc, run_label)])

    styles = _build_styles()
    mono   = styles["Mono"]
    elems  = []

    # ── Cover / summary ───────────────────────────────────────────────────────
    elems += [
        Spacer(1, 0.3 * inch),
        Paragraph(f"VICReg-WNAE Training Report", styles["RunTitle"]),
        Paragraph(run_label, styles["SubTitle"]),
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Caption"],
        ),
        Spacer(1, 0.25 * inch),
        HRFlowable(width="100%", thickness=1, color=TEAL),
        Spacer(1, 0.15 * inch),
    ]

    # Best-epoch badge
    if best_epoch is not None:
        badge_data = [[Paragraph(
            f"<b><font color='white' size=13>★  Best epoch: {best_epoch}</font></b>",
            styles["Normal"],
        )]]
        badge = Table(badge_data, colWidths=[6.5 * inch])
        badge.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elems += [badge, Spacer(1, 0.2 * inch)]

    # Metrics summary table from CSV
    if rows:
        best_row  = rows[best_epoch] if best_epoch is not None and best_epoch < len(rows) else None
        final_row = rows[-1]
        best_auc_row = max(rows, key=lambda r: float(r["auc"]))

        def fmt(r, key): return f"{float(r[key]):.4f}" if r else "—"

        summary_header = [
            Paragraph("<b>Metric</b>",       mono),
            Paragraph("<b>Best Epoch</b>",   mono),
            Paragraph("<b>Final Epoch</b>",  mono),
            Paragraph("<b>Best Overall</b>", mono),
        ]
        summary_rows = [
            summary_header,
            [Paragraph("Epoch", mono),
             Paragraph(str(best_epoch) if best_epoch is not None else "—", mono),
             Paragraph(final_row["epoch"], mono),
             Paragraph(best_auc_row["epoch"], mono)],
            [Paragraph("Train loss", mono),
             Paragraph(fmt(best_row, "training_loss"), mono),
             Paragraph(fmt(final_row, "training_loss"), mono),
             Paragraph("—", mono)],
            [Paragraph("Val loss", mono),
             Paragraph(fmt(best_row, "validation_loss"), mono),
             Paragraph(fmt(final_row, "validation_loss"), mono),
             Paragraph("—", mono)],
            [Paragraph("AUC", mono),
             Paragraph(fmt(best_row, "auc"), mono),
             Paragraph(fmt(final_row, "auc"), mono),
             Paragraph(fmt(best_auc_row, "auc"), mono)],
        ]

        col_w = [2.0 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
        summary_tbl = Table(summary_rows, colWidths=col_w)
        summary_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
            ("GRID",          (0, 0), (-1, -1), 0.4, MID),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (0, -1),  8),
        ]))
        elems += [*section_heading("Key Metrics", styles), summary_tbl, Spacer(1, 0.2 * inch)]

    # Training history plot
    elems += section_heading("Loss & AUC History", styles)
    if training_loss_path.exists():
        elems += [
            Image(str(training_loss_path), width=6.5 * inch, height=4.0 * inch),
            Spacer(1, 4),
            Paragraph("Training loss, validation loss, and AUC over epochs.", styles["Caption"]),
        ]
    else:
        elems.append(Paragraph("train_history.png not found.", styles["Normal"]))

    # ── Config page ───────────────────────────────────────────────────────────
    elems.append(PageBreak())
    elems += section_heading("Configuration", styles)
    if config:
        _add_config(config, elems, mono)
    else:
        elems.append(Paragraph("No configuration provided.", styles["Normal"]))

    # ── Feature histogram pages ───────────────────────────────────────────────
    if feature_dirs and epoch_order:
        elems.append(PageBreak())
        elems += section_heading("Feature Histograms", styles)

        num_cols = 3
        col_w    = 2.1 * inch

        for epoch in epoch_order:
            is_best = (epoch == best_epoch)
            heading_style = styles["BestEpochH"] if is_best else styles["EpochH"]
            label = f"★  Best Epoch {epoch}" if is_best else f"Epoch {epoch}"

            # Collect images into rows
            cells, row = [], []
            for i, fdir in enumerate(feature_dirs):
                img_path = _epoch_plot(fdir, epoch)
                if img_path:
                    cell = [
                        Image(str(img_path), width=col_w, height=1.65 * inch),
                        Paragraph(fdir.name, styles["Caption"]),
                    ]
                else:
                    cell = [Paragraph(f"[missing] {fdir.name}", styles["Caption"])]
                row.append(cell)
                if len(row) == num_cols:
                    cells.append(row)
                    row = []
            if row:
                while len(row) < num_cols:
                    row.append([Paragraph("", styles["Normal"])])
                cells.append(row)

            tbl = Table(cells, colWidths=[col_w] * num_cols)
            tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                *(
                    [("BOX", (0, 0), (-1, -1), 1.5, GOLD),
                     ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb"))]
                    if is_best else []
                ),
            ]))

            block = [Paragraph(label, heading_style), tbl, Spacer(1, 8)]
            elems.append(KeepTogether(block))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(elems)
    print(f"Report saved → {pdf_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Generate PDF report for a VICReg-WNAE run.")
    p.add_argument("output_dir", nargs="?", default="output_vicreg_wnae",
                   help="Path to the training output directory")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (auto-detected if omitted)")
    p.add_argument("--interval", type=int, default=10,
                   help="Epoch interval for feature histogram pages (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = Path(args.output_dir)

    # Auto-detect config
    cfg = None
    config_path = Path(args.config) if args.config else None
    if config_path is None:
        candidates = [
            Path("config/vicreg_wnae_config.yaml"),
            Path("config/config.yaml"),
        ]
        config_path = next((c for c in candidates if c.exists()), None)

    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Using config: {config_path}")
    else:
        print("No config file found — skipping config section.")

    create_report(output_dir, config=cfg, interval=args.interval)
