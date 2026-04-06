"""
Apple Orchard Doctor System - Flask Backend
Run: python app.py
Requires: pip install flask flask-mysqldb flask-bcrypt flask-session
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import os
import uuid
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from werkzeug.utils import secure_filename
from cnn.predictor import predict as cnn_predict, db_disease_name, is_ready as cnn_ready

UPLOAD_FOLDER  = os.path.join("static", "uploads")
ALLOWED_EXT    = {"png", "jpg", "jpeg", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "orchard-secret-2024-change-in-production")

# ── Database Config ──────────────────────────────────────────────────────────
app.config["MYSQL_HOST"]     = os.environ.get("DB_HOST", "localhost")
app.config["MYSQL_USER"]     = os.environ.get("DB_USER", "root")
app.config["MYSQL_PASSWORD"] = os.environ.get("DB_PASSWORD", "1234")
app.config["MYSQL_DB"]       = os.environ.get("DB_NAME", "Apple_orchard_doctor")
app.config["MYSQL_CURSORCLASS"] = "DictCursor"
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # 8 MB max upload

mysql  = MySQL(app)
bcrypt = Bcrypt(app)


# ── Helpers ──────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def expert_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") not in ("expert", "manager"):
            flash("Expert access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


# ── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form["name"].strip()
        email    = request.form["email"].strip().lower()
        password = request.form["password"]
        role     = request.form.get("role", "farmer")
        phone    = request.form.get("phone", "")
        location = request.form.get("location", "")

        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        cur = mysql.connection.cursor()
        try:
            cur.execute(
                "INSERT INTO users (name, email, password, role, phone, location) VALUES (%s,%s,%s,%s,%s,%s)",
                (name, email, hashed, role, phone, location)
            )
            mysql.connection.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash("Email already registered.", "danger")
        finally:
            cur.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form["email"].strip().lower()
        password = request.form["password"]
        role     = request.form.get("role", "farmer")

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            if user["role"] != role:
                flash(f"Incorrect role. Your account is registered as '{user['role'].capitalize()}'.", "danger")
            else:
                session["user_id"] = user["id"]
                session["name"]    = user["name"]
                session["role"]    = user["role"]
                flash(f"Welcome back, {user['name']}!", "success")
                return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM diseases")
    disease_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM diagnoses WHERE user_id = %s", (session["user_id"],))
    my_diagnoses  = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM advice")
    advice_count  = cur.fetchone()["cnt"]
    cur.execute("""
        SELECT di.id AS diagnosis_id, d.id AS disease_id, d.name AS disease, di.diagnosed_at
        FROM diagnoses di JOIN diseases d ON d.id = di.disease_id
        WHERE di.user_id = %s ORDER BY di.diagnosed_at DESC LIMIT 5
    """, (session["user_id"],))
    recent = cur.fetchall()
    cur.close()
    return render_template("dashboard.html",
        disease_count=disease_count, my_diagnoses=my_diagnoses,
        advice_count=advice_count, recent=recent)


# ── PDF helpers ──────────────────────────────────────────────────────────────
C_GREEN      = "#2d6a4f"
C_DARK_GREEN = "#1b4332"
C_MID_GREEN  = "#40916c"
C_LIGHT_GREEN= "#95d5b2"
C_PALE_GREEN = "#d8f3dc"
C_TEXT       = "#1a1a2e"
C_SUBTEXT    = "#4a4a6a"
C_RED        = "#c0392b"
C_ORANGE     = "#d35400"
C_WARN_BG    = "#fff3cd"
C_WARN_BORDER= "#e0a800"

def _draw_bg(canvas, doc):
    """Full-page apple leaf watermark background."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    W, H = doc.pagesize
    canvas.saveState()

    # Very light green page tint
    canvas.setFillColor(colors.HexColor("#f6fff8"))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # ── Large central leaf ──
    canvas.setFillColor(colors.HexColor(C_GREEN))
    canvas.setFillAlpha(0.045)
    cx, cy = W / 2, H / 2
    lw, lh = 11*cm, 18*cm
    p = canvas.beginPath()
    p.moveTo(cx, cy - lh/2)
    p.curveTo(cx + lw*0.85, cy - lh*0.25, cx + lw*0.9, cy + lh*0.25, cx, cy + lh/2)
    p.curveTo(cx - lw*0.9,  cy + lh*0.25, cx - lw*0.85,cy - lh*0.25, cx, cy - lh/2)
    canvas.drawPath(p, fill=1, stroke=0)

    # Midrib
    canvas.setStrokeColor(colors.HexColor(C_DARK_GREEN))
    canvas.setStrokeAlpha(0.07)
    canvas.setLineWidth(5)
    canvas.line(cx, cy - lh/2, cx, cy + lh/2)

    # Veins
    canvas.setLineWidth(2)
    for i in range(1, 7):
        t  = i / 7
        vy = cy - lh/2 + lh * t
        vx = lw * 0.52 * (1 - abs(t - 0.5) * 1.6)
        canvas.line(cx, vy, cx + vx, vy + lh*0.06)
        canvas.line(cx, vy, cx - vx, vy + lh*0.06)

    # ── Small corner leaf accents ──
    canvas.setFillAlpha(0.055)
    canvas.setStrokeAlpha(0)
    for (lcx, lcy, lscale, angle) in [
        (W*0.08, H*0.92, 0.45, 30),
        (W*0.92, H*0.08, 0.38, -150),
        (W*0.90, H*0.88, 0.30, 60),
        (W*0.10, H*0.12, 0.28, -60),
    ]:
        canvas.saveState()
        canvas.translate(lcx, lcy)
        canvas.rotate(angle)
        slw, slh = lw*lscale, lh*lscale
        sp = canvas.beginPath()
        sp.moveTo(0, -slh/2)
        sp.curveTo( slw*0.85, -slh*0.25,  slw*0.9,  slh*0.25, 0,  slh/2)
        sp.curveTo(-slw*0.9,   slh*0.25, -slw*0.85, -slh*0.25, 0, -slh/2)
        canvas.drawPath(sp, fill=1, stroke=0)
        canvas.restoreState()

    canvas.restoreState()

def _pdf_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    s = getSampleStyleSheet()
    return s, {
        "h2":      ParagraphStyle("h2",      fontName="Helvetica-Bold", fontSize=14, textColor=colors.HexColor(C_DARK_GREEN), spaceBefore=10, spaceAfter=4),
        "h3":      ParagraphStyle("h3",      fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor(C_GREEN),      spaceBefore=6,  spaceAfter=3),
        "body":    ParagraphStyle("body",    fontName="Helvetica",      fontSize=9,  leading=14, textColor=colors.HexColor(C_TEXT),       spaceAfter=4),
        "small":   ParagraphStyle("small",   fontName="Helvetica",      fontSize=8,  leading=12, textColor=colors.HexColor(C_SUBTEXT),    spaceAfter=2),
        "bullet":  ParagraphStyle("bullet",  fontName="Helvetica",      fontSize=9,  leading=13, textColor=colors.HexColor(C_TEXT),       leftIndent=14, spaceAfter=2),
        "warn":    ParagraphStyle("warn",    fontName="Helvetica-Bold", fontSize=9,  textColor=colors.HexColor(C_RED)),
        "cell":    ParagraphStyle("cell",    fontName="Helvetica",      fontSize=8,  leading=11, textColor=colors.HexColor(C_TEXT)),
        "cellhdr": ParagraphStyle("cellhdr", fontName="Helvetica-Bold", fontSize=8,  textColor=colors.white),
        "sev_high":     ParagraphStyle("sevH", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor(C_RED)),
        "sev_critical": ParagraphStyle("sevC", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#7b241c")),
        "sev_medium":   ParagraphStyle("sevM", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor(C_ORANGE)),
        "sev_low":      ParagraphStyle("sevL", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor(C_GREEN)),
    }

def _sev_style(extra, severity):
    return extra.get(f"sev_{severity}", extra["body"])

def _section_box(canvas, x, y, w, h, title, icon=""):
    """Draw a labelled section box border with a green title bar."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    canvas.saveState()
    # Box border
    canvas.setStrokeColor(colors.HexColor(C_LIGHT_GREEN))
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, y, w, h, 4, fill=0, stroke=1)
    # Title bar
    canvas.setFillColor(colors.HexColor(C_GREEN))
    canvas.roundRect(x, y + h - 0.65*cm, w, 0.65*cm, 4, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(x + 0.3*cm, y + h - 0.48*cm, f"{icon}  {title}" if icon else title)
    canvas.restoreState()

def _treatment_table(treatments, extra):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    TYPE_COLORS = {
        "chemical":   colors.HexColor("#fdecea"),
        "organic":    colors.HexColor("#e8f5e9"),
        "preventive": colors.HexColor("#e3f2fd"),
        "cultural":   colors.HexColor("#fff8e1"),
    }
    TYPE_ICONS = {"chemical": "[Rx]", "organic": "[Org]", "preventive": "[Prev]", "cultural": "[Cult]"}
    hdr = [
        Paragraph("Type",           extra["cellhdr"]),
        Paragraph("Medicine / Name",extra["cellhdr"]),
        Paragraph("Dosage",         extra["cellhdr"]),
        Paragraph("Instructions",   extra["cellhdr"]),
        Paragraph("Timing",         extra["cellhdr"]),
        Paragraph("Est. Cost",      extra["cellhdr"]),
    ]
    rows = [hdr]
    row_colors = []
    for i, t in enumerate(treatments, start=1):
        icon = TYPE_ICONS.get(t["type"], "")
        rows.append([
            Paragraph(f"{icon} {t['type'].capitalize()}", extra["cell"]),
            Paragraph(t["name"],                          extra["cell"]),
            Paragraph(t["dosage"] or "—",                 extra["cell"]),
            Paragraph(t["instructions"] or "—",           extra["cell"]),
            Paragraph(t["timing"] or "—",                 extra["cell"]),
            Paragraph(t["cost_estimate"] or "—",          extra["cell"]),
        ])
        row_colors.append(("BACKGROUND", (0,i), (-1,i), TYPE_COLORS.get(t["type"], colors.HexColor("#f0f7f4"))))
    tbl = Table(rows, colWidths=[2.0*cm, 3.8*cm, 2.4*cm, 5.5*cm, 2.8*cm, 2.0*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor(C_DARK_GREEN)),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("LEADING",       (0,0), (-1,-1), 11),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#c8e6c9")),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f1f8f4")]),
    ] + row_colors))
    return tbl

def _draw_page(canvas, doc, title, subtitle, farmer, date_str=None):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    W, H = doc.pagesize

    _draw_bg(canvas, doc)
    canvas.saveState()

    # ── Header band ──
    canvas.setFillColor(colors.HexColor(C_DARK_GREEN))
    canvas.rect(0, H - 4.6*cm, W, 4.6*cm, fill=1, stroke=0)
    # Lighter overlay left half
    canvas.setFillColor(colors.HexColor(C_MID_GREEN))
    canvas.setFillAlpha(0.30)
    canvas.rect(0, H - 4.6*cm, W*0.6, 4.6*cm, fill=1, stroke=0)
    canvas.setFillAlpha(1)
    # Left accent stripe
    canvas.setFillColor(colors.HexColor(C_LIGHT_GREEN))
    canvas.rect(0, H - 4.6*cm, 0.45*cm, 4.6*cm, fill=1, stroke=0)
    # Decorative leaf in header top-right
    canvas.setFillColor(colors.HexColor(C_LIGHT_GREEN))
    canvas.setFillAlpha(0.15)
    lx, ly = W - 2.2*cm, H - 2.3*cm
    lp = canvas.beginPath()
    lp.moveTo(lx, ly - 1.6*cm)
    lp.curveTo(lx+1.3*cm, ly-0.5*cm, lx+1.3*cm, ly+0.5*cm, lx, ly+1.6*cm)
    lp.curveTo(lx-1.3*cm, ly+0.5*cm, lx-1.3*cm, ly-0.5*cm, lx, ly-1.6*cm)
    canvas.drawPath(lp, fill=1, stroke=0)
    canvas.setFillAlpha(1)
    # Title text
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(0.9*cm, H - 1.7*cm, title)
    # Subtitle
    canvas.setFont("Helvetica", 10)
    canvas.setFillColor(colors.HexColor(C_PALE_GREEN))
    canvas.drawString(0.9*cm, H - 2.45*cm, subtitle)
    # Farmer + date row
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#b7e4c7"))
    info = f"  Farmer: {farmer}"
    if date_str:
        info += f"     |     Date & Time: {date_str}"
    canvas.drawString(0.9*cm, H - 3.15*cm, info)
    # Bottom border of header
    canvas.setStrokeColor(colors.HexColor(C_LIGHT_GREEN))
    canvas.setLineWidth(1.5)
    canvas.line(0, H - 4.6*cm, W, H - 4.6*cm)

    # ── Footer ──
    canvas.setFillColor(colors.HexColor(C_DARK_GREEN))
    canvas.rect(0, 0, W, 1.5*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(C_LIGHT_GREEN))
    canvas.rect(0, 1.5*cm, W, 0.1*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(C_PALE_GREEN))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(0.9*cm, 0.5*cm, "Generated by Leaf Orchard Doctor System  —  Kashmir Valley")
    canvas.drawRightString(W - 0.9*cm, 0.5*cm, f"Page {doc.page}")

    canvas.restoreState()


# ── Dashboard PDF Download ───────────────────────────────────────────────────
@app.route("/dashboard/pdf")
@login_required
def dashboard_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
    from reportlab.lib.units import cm
    import io, datetime

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT di.diagnosed_at, di.notes, d.id AS disease_id, d.name AS disease,
               d.severity, d.description, d.cause
        FROM diagnoses di JOIN diseases d ON d.id = di.disease_id
        WHERE di.user_id = %s ORDER BY di.diagnosed_at DESC
    """, (session["user_id"],))
    diagnoses = cur.fetchall()

    disease_ids = list({r["disease_id"] for r in diagnoses})
    treatments_map = {}
    symptoms_map   = {}
    if disease_ids:
        fmt = ",".join(["%s"] * len(disease_ids))
        cur.execute(f"SELECT * FROM treatments WHERE disease_id IN ({fmt}) ORDER BY disease_id, type", disease_ids)
        for t in cur.fetchall():
            treatments_map.setdefault(t["disease_id"], []).append(t)
        cur.execute(f"SELECT ds.disease_id, s.name FROM symptoms s JOIN disease_symptoms ds ON ds.symptom_id=s.id WHERE ds.disease_id IN ({fmt})", disease_ids)
        for s in cur.fetchall():
            symptoms_map.setdefault(s["disease_id"], []).append(s["name"])
    cur.close()

    s, extra = _pdf_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=5.0*cm, bottomMargin=2.0*cm)
    now_str = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

    def on_page(canvas, doc):
        _draw_page(canvas, doc,
                   "Leaf Orchard Doctor",
                   "Apple Leaf Disease Report  |  Full Diagnosis History",
                   session["name"], now_str)

    story = []
    if not diagnoses:
        story.append(Paragraph("No diagnoses logged yet.", extra["body"]))
    else:
        seen = set()
        for r in diagnoses:
            date_str = r["diagnosed_at"].strftime("%d %b %Y, %I:%M %p")
            did = r["disease_id"]
            sev = r["severity"]
            SEV_COLOR = {"critical": C_RED, "high": C_RED, "medium": C_ORANGE, "low": C_GREEN}
            sev_hex = SEV_COLOR.get(sev, C_TEXT)

            block = []
            # ── Disease name + severity badge ──
            block.append(HRFlowable(width="100%", thickness=1.2,
                                    color=colors.HexColor(C_GREEN), spaceAfter=6))
            block.append(Paragraph(
                f"{r['disease']}  "
                f"<font size=8 color='{sev_hex}'>[{sev.upper()}]</font>",
                extra["h2"]))
            block.append(Paragraph(
                f"Diagnosed: {date_str}",
                extra["small"]))

            # ── Description + Cause ──
            if r.get("description"):
                block.append(Spacer(1, 0.15*cm))
                block.append(Paragraph("<b>[i] About this Disease</b>", extra["h3"]))
                block.append(Paragraph(r["description"], extra["body"]))
            if r.get("cause"):
                block.append(Paragraph(f"<b>[!] Cause:</b> {r['cause']}", extra["body"]))
            if r.get("notes") and did not in seen:
                block.append(Paragraph(f"<b>[*] Notes:</b> {r['notes']}", extra["body"]))

            # ── Symptoms bullets (first occurrence only) ──
            if did not in seen and did in symptoms_map:
                block.append(Spacer(1, 0.15*cm))
                block.append(Paragraph("<b>[~] Symptoms</b>", extra["h3"]))
                for sym in symptoms_map[did]:
                    block.append(Paragraph(f"\u2022  {sym}", extra["bullet"]))

            # ── Treatments (first occurrence only) ──
            if did not in seen and did in treatments_map:
                block.append(Spacer(1, 0.2*cm))
                block.append(Paragraph("<b>[+] Recommended Treatments</b>", extra["h3"]))
                block.append(Spacer(1, 0.1*cm))
                block.append(_treatment_table(treatments_map[did], extra))

            seen.add(did)
            block.append(Spacer(1, 0.5*cm))
            story.append(KeepTogether(block))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    response = make_response(buf.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=diagnosis_report.pdf"
    return response


# ── Single Diagnosis PDF ─────────────────────────────────────────────────────
@app.route("/diagnosis/<int:diagnosis_id>/pdf")
@login_required
def diagnosis_pdf(diagnosis_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.units import cm
    import io

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT di.diagnosed_at, di.notes, d.id AS disease_id, d.name AS disease,
               d.severity, d.description, d.cause
        FROM diagnoses di JOIN diseases d ON d.id = di.disease_id
        WHERE di.id = %s AND di.user_id = %s
    """, (diagnosis_id, session["user_id"]))
    row = cur.fetchone()
    if not row:
        flash("Diagnosis not found.", "danger")
        return redirect(url_for("dashboard"))

    cur.execute("SELECT * FROM treatments WHERE disease_id = %s ORDER BY type", (row["disease_id"],))
    treatments = cur.fetchall()
    cur.execute("""
        SELECT s.name FROM symptoms s
        JOIN disease_symptoms ds ON ds.symptom_id = s.id
        WHERE ds.disease_id = %s
    """, (row["disease_id"],))
    symptoms = [r["name"] for r in cur.fetchall()]
    cur.close()

    s, extra = _pdf_styles()
    date_str = row["diagnosed_at"].strftime("%d %b %Y, %I:%M %p")
    sev      = row["severity"]
    SEV_COLOR = {"critical": C_RED, "high": C_RED, "medium": C_ORANGE, "low": C_GREEN}
    sev_hex  = SEV_COLOR.get(sev, C_TEXT)

    # Parse confidence from notes if CNN-generated
    confidence_line = ""
    if row.get("notes") and "confidence" in (row["notes"] or "").lower():
        confidence_line = row["notes"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=5.0*cm, bottomMargin=2.0*cm)

    def on_page(canvas, doc):
        _draw_page(canvas, doc,
                   "Leaf Orchard Doctor",
                   "Apple Leaf Disease Report",
                   session["name"], date_str)

    story = []

    # ── SECTION 1: Disease Info ──
    story.append(Paragraph(
        f"{row['disease']}  "
        f"<font size=9 color='{sev_hex}'>[{sev.upper()}]</font>",
        extra["h2"]))
    if confidence_line:
        story.append(Paragraph(
            f"<font color='{C_GREEN}'>[CNN]</font>  {confidence_line}",
            extra["small"]))
    story.append(Spacer(1, 0.1*cm))
    story.append(HRFlowable(width="100%", thickness=0.8,
                            color=colors.HexColor(C_LIGHT_GREEN), spaceAfter=8))

    if row.get("description"):
        story.append(Paragraph("<b>[i] About this Disease</b>", extra["h3"]))
        story.append(Paragraph(row["description"], extra["body"]))

    if row.get("cause"):
        story.append(Paragraph("<b>[!] Cause</b>", extra["h3"]))
        story.append(Paragraph(
            f"<font color='{C_RED}'>⚠</font>  {row['cause']}",
            extra["body"]))

    # ── SECTION 2: Symptoms ──
    if symptoms:
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor(C_LIGHT_GREEN), spaceAfter=6))
        story.append(Paragraph("<b>[~] Symptoms</b>", extra["h3"]))
        for sym in symptoms:
            story.append(Paragraph(f"\u2022  {sym}", extra["bullet"]))

    # ── SECTION 3: Treatments ──
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor(C_LIGHT_GREEN), spaceAfter=6))
    story.append(Paragraph("<b>[+] Recommended Treatments</b>", extra["h3"]))
    story.append(Spacer(1, 0.1*cm))
    if treatments:
        story.append(_treatment_table(treatments, extra))
    else:
        story.append(Paragraph(
            "<font color='#c0392b'>No treatments found for this disease.</font>",
            extra["body"]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    response = make_response(buf.read())
    response.headers["Content-Type"] = "application/pdf"
    safe_name = row["disease"].replace(" ", "_")
    response.headers["Content-Disposition"] = f"attachment; filename={safe_name}_report.pdf"
    return response


# ── Disease Identification ────────────────────────────────────────────────────
@app.route("/identify", methods=["GET", "POST"])
@login_required
def identify():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM symptoms ORDER BY body_part, name")
    symptoms = cur.fetchall()

    results = []
    selected_ids = []

    if request.method == "POST":
        selected_ids = request.form.getlist("symptoms", type=int)
        if selected_ids:
            fmt = ",".join(["%s"] * len(selected_ids))
            cur.execute(f"""
                SELECT d.*, c.name AS category,
                       COUNT(ds.symptom_id) AS matched,
                       (COUNT(ds.symptom_id) * 100.0 /
                           (SELECT COUNT(*) FROM disease_symptoms WHERE disease_id = d.id)
                       ) AS match_pct
                FROM diseases d
                JOIN categories c ON c.id = d.category_id
                JOIN disease_symptoms ds ON ds.disease_id = d.id
                WHERE ds.symptom_id IN ({fmt})
                GROUP BY d.id
                HAVING matched >= 1
                ORDER BY match_pct DESC, matched DESC
                LIMIT 5
            """, tuple(selected_ids))
            results = cur.fetchall()

    cur.close()
    return render_template("identify.html", symptoms=symptoms,
                           results=results, selected_ids=selected_ids)


# ── Disease Detail + Treatment ────────────────────────────────────────────────
@app.route("/disease/<int:disease_id>")
@login_required
def disease_detail(disease_id):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT d.*, c.name AS category
        FROM diseases d JOIN categories c ON c.id = d.category_id
        WHERE d.id = %s
    """, (disease_id,))
    disease = cur.fetchone()
    if not disease:
        flash("Disease not found.", "danger")
        return redirect(url_for("diseases"))

    cur.execute("""
        SELECT s.* FROM symptoms s
        JOIN disease_symptoms ds ON ds.symptom_id = s.id
        WHERE ds.disease_id = %s
    """, (disease_id,))
    symptoms = cur.fetchall()

    cur.execute("SELECT * FROM treatments WHERE disease_id = %s ORDER BY type", (disease_id,))
    treatments = cur.fetchall()

    cur.execute("""
        SELECT a.*, u.name AS expert_name FROM advice a
        JOIN users u ON u.id = a.expert_id
        WHERE a.disease_id = %s ORDER BY a.created_at DESC
    """, (disease_id,))
    advice_list = cur.fetchall()
    cur.close()

    return render_template("disease_detail.html",
        disease=disease, symptoms=symptoms, treatments=treatments, advice=advice_list)


# ── Disease Database ──────────────────────────────────────────────────────────
@app.route("/diseases")
@login_required
def diseases():
    search   = request.args.get("q", "")
    category = request.args.get("category", "")
    severity = request.args.get("severity", "")

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM categories ORDER BY name")
    categories = cur.fetchall()

    query  = "SELECT d.*, c.name AS category FROM diseases d JOIN categories c ON c.id=d.category_id WHERE 1=1"
    params = []
    if search:
        query += " AND (d.name LIKE %s OR d.description LIKE %s)"
        params += [f"%{search}%", f"%{search}%"]
    if category:
        query += " AND d.category_id = %s"
        params.append(category)
    if severity:
        query += " AND d.severity = %s"
        params.append(severity)
    query += " ORDER BY d.severity DESC, d.name"

    cur.execute(query, params)
    disease_list = cur.fetchall()
    cur.close()

    return render_template("diseases.html",
        diseases=disease_list, categories=categories,
        search=search, selected_category=category, selected_severity=severity)


# ── Delete a Diagnosis ───────────────────────────────────────────────────────
@app.route("/diagnosis/<int:diagnosis_id>/delete", methods=["POST"])
@login_required
def delete_diagnosis(diagnosis_id):
    cur = mysql.connection.cursor()
    cur.execute(
        "DELETE FROM diagnoses WHERE id = %s AND user_id = %s",
        (diagnosis_id, session["user_id"])
    )
    mysql.connection.commit()
    cur.close()
    flash("Diagnosis deleted.", "success")
    return redirect(url_for("dashboard"))


# ── Log a Diagnosis ───────────────────────────────────────────────────────────
@app.route("/log-diagnosis", methods=["POST"])
@login_required
def log_diagnosis():
    disease_id = request.form.get("disease_id", type=int)
    notes      = request.form.get("notes", "")
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO diagnoses (user_id, disease_id, notes) VALUES (%s, %s, %s)",
        (session["user_id"], disease_id, notes)
    )
    mysql.connection.commit()
    cur.close()
    flash("Diagnosis logged successfully.", "success")
    return redirect(url_for("disease_detail", disease_id=disease_id))


# ── Orchard Manager Role Page ────────────────────────────────────────────
@app.route("/manager-role")
@login_required
def manager_role():
    cur = mysql.connection.cursor()
    # pruning logs
    cur.execute("SELECT * FROM pruning_logs WHERE user_id=%s ORDER BY prune_date DESC", (session["user_id"],))
    prune_logs = cur.fetchall()
    # checklist
    cur.execute("SELECT task_text, is_done FROM checklist_progress WHERE user_id=%s", (session["user_id"],))
    checklist_rows = {r["task_text"]: r["is_done"] for r in cur.fetchall()}
    # tasks
    cur.execute("SELECT * FROM tasks WHERE manager_id=%s ORDER BY created_at DESC", (session["user_id"],))
    tasks = cur.fetchall()
    # stats
    cur.execute("SELECT COUNT(*) AS cnt FROM diagnoses WHERE user_id=%s", (session["user_id"],))
    total_diagnoses = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM diagnoses WHERE user_id=%s AND MONTH(diagnosed_at)=MONTH(NOW()) AND YEAR(diagnosed_at)=YEAR(NOW())", (session["user_id"],))
    recent_count = cur.fetchone()["cnt"]
    cur.close()
    return render_template("manager_role.html",
        prune_logs=prune_logs, checklist_rows=checklist_rows,
        tasks=tasks, total_diagnoses=total_diagnoses, recent_count=recent_count)


# ── Pruning Log Routes ─────────────────────────────────────────────────────
@app.route("/pruning/add", methods=["POST"])
@login_required
def add_prune_log():
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO pruning_logs (user_id, prune_date, area, trees_pruned) VALUES (%s,%s,%s,%s)",
        (session["user_id"], request.form["prune_date"], request.form["area"], request.form["trees_pruned"])
    )
    mysql.connection.commit()
    cur.close()
    flash("Pruning activity logged.", "success")
    return redirect(url_for("manager_role") + "#section-prune")


@app.route("/pruning/delete/<int:log_id>", methods=["POST"])
@login_required
def delete_prune_log(log_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM pruning_logs WHERE id=%s AND user_id=%s", (log_id, session["user_id"]))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for("manager_role") + "#section-prune")


# ── Checklist Routes ─────────────────────────────────────────────────────────
@app.route("/checklist/toggle", methods=["POST"])
@login_required
def toggle_checklist():
    task_text = request.json.get("task_text")
    is_done   = request.json.get("is_done", 0)
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO checklist_progress (user_id, task_text, is_done)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE is_done=%s
    """, (session["user_id"], task_text, is_done, is_done))
    mysql.connection.commit()
    cur.close()
    return jsonify({"ok": True})


@app.route("/checklist/reset", methods=["POST"])
@login_required
def reset_checklist():
    cur = mysql.connection.cursor()
    cur.execute("UPDATE checklist_progress SET is_done=0 WHERE user_id=%s", (session["user_id"],))
    mysql.connection.commit()
    cur.close()
    return jsonify({"ok": True})


# ── Task (Kanban) Routes ────────────────────────────────────────────────────
@app.route("/tasks/add", methods=["POST"])
@login_required
def add_task():
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO tasks (manager_id, worker_name, task_type, task_date) VALUES (%s,%s,%s,%s)",
        (session["user_id"], request.form["worker_name"], request.form["task_type"], request.form["task_date"])
    )
    mysql.connection.commit()
    cur.close()
    flash("Task assigned.", "success")
    return redirect(url_for("manager_role") + "#section-team")


@app.route("/tasks/advance/<int:task_id>", methods=["POST"])
@login_required
def advance_task(task_id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE tasks SET stage=LEAST(stage+1,2) WHERE id=%s AND manager_id=%s", (task_id, session["user_id"]))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for("manager_role") + "#section-team")


@app.route("/tasks/delete/<int:task_id>", methods=["POST"])
@login_required
def delete_task(task_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM tasks WHERE id=%s AND manager_id=%s", (task_id, session["user_id"]))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for("manager_role") + "#section-team")


# ── Expert Advice ─────────────────────────────────────────────────────────────
@app.route("/advice")
@login_required
def advice_list():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT a.*, u.name AS expert_name, d.name AS disease_name
        FROM advice a
        JOIN users u ON u.id = a.expert_id
        LEFT JOIN diseases d ON d.id = a.disease_id
        ORDER BY a.created_at DESC
    """)
    items = cur.fetchall()
    cur.close()
    return render_template("advice.html", items=items)


@app.route("/advice/new", methods=["GET", "POST"])
@login_required
@expert_required
def new_advice():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, name FROM diseases ORDER BY name")
    diseases = cur.fetchall()

    if request.method == "POST":
        title      = request.form["title"]
        content    = request.form["content"]
        disease_id = request.form.get("disease_id") or None
        cur.execute(
            "INSERT INTO advice (expert_id, disease_id, title, content) VALUES (%s,%s,%s,%s)",
            (session["user_id"], disease_id, title, content)
        )
        mysql.connection.commit()
        cur.close()
        flash("Advice posted successfully.", "success")
        return redirect(url_for("advice_list"))

    cur.close()
    return render_template("new_advice.html", diseases=diseases)


# ── CNN: Leaf Scan ───────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/scan", methods=["GET", "POST"])
@login_required
def scan():
    """Upload a leaf photo → CNN predicts disease → show treatments."""
    result      = None
    disease     = None
    treatments  = []
    image_url   = None
    error       = None
    model_ready = cnn_ready()

    if request.method == "POST":
        if not model_ready:
            error = "The CNN model has not been trained yet. Run  python cnn/train.py  first."
        elif "leaf_image" not in request.files or request.files["leaf_image"].filename == "":
            error = "Please select a leaf image to upload."
        else:
            file = request.files["leaf_image"]
            if not allowed_file(file.filename):
                error = "Invalid file type. Please upload a JPG, PNG, or WEBP image."
            else:
                # Save with a unique name to avoid collisions
                ext      = file.filename.rsplit(".", 1)[1].lower()
                filename = f"{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                image_url = url_for("static", filename=f"uploads/{filename}")

                try:
                    result = cnn_predict(filepath)

                    if not result["is_healthy"]:
                        db_name = result.get("db_name")
                        if db_name:
                            cur = mysql.connection.cursor()
                            cur.execute("""
                                SELECT d.*, c.name AS category
                                FROM diseases d
                                JOIN categories c ON c.id = d.category_id
                                WHERE d.name = %s
                            """, (db_name,))
                            disease = cur.fetchone()

                            if disease:
                                cur.execute(
                                    "SELECT * FROM treatments WHERE disease_id = %s ORDER BY type",
                                    (disease["id"],)
                                )
                                treatments = cur.fetchall()

                                # Auto-log the diagnosis
                                cur.execute(
                                    "INSERT INTO diagnoses (user_id, disease_id, notes) VALUES (%s, %s, %s)",
                                    (session["user_id"], disease["id"], f"Detected by CNN ({result['confidence']}% confidence)")
                                )
                                mysql.connection.commit()
                            cur.close()

                except Exception as e:
                    error = f"Prediction failed: {str(e)}"

    return render_template("scan.html",
        result=result, disease=disease, treatments=treatments,
        image_url=image_url, error=error, model_ready=model_ready)


# ── API: symptoms by body part (used by JS) ───────────────────────────────────
@app.route("/api/symptoms")
def api_symptoms():
    part = request.args.get("part", "")
    cur  = mysql.connection.cursor()
    if part:
        cur.execute("SELECT * FROM symptoms WHERE body_part=%s ORDER BY name", (part,))
    else:
        cur.execute("SELECT * FROM symptoms ORDER BY body_part, name")
    data = cur.fetchall()
    cur.close()
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=True, reloader_type='stat')