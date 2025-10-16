from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ================= Helpers =================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_table(df: pd.DataFrame, path: str, index: bool = False):
    df.to_csv(path, index=index, encoding="utf-8-sig")

def build_export_links(base_dir, scope_name, features, freq_tables_clean, like_tables_raw, prior_df):
    """
    - freq_tables_clean: sudah reset_index (kolom 1 = nama fitur) -> simpan index=False
    - like_tables_raw:  DataFrame likelihood dengan index nilai + 'All'
                        -> kita reset_index agar kolom 1 = nama fitur lalu simpan index=False
    """
    scope_dir = os.path.join(base_dir, scope_name)
    ensure_dir(scope_dir)
    links = {"prior": None, "freq": [], "like": []}

    # prior
    prior_csv = os.path.join(scope_dir, f"prior_{scope_name}.csv")
    prior_df.to_csv(prior_csv, index=False, encoding="utf-8-sig")
    rel_prior = os.path.relpath(prior_csv, os.path.join(app.config["UPLOAD_FOLDER"], "exports"))
    links["prior"] = url_for("download_export", subpath=rel_prior.replace("\\", "/"))

    # freq & like per fitur
    for feat in features:
        # Frequency (sudah bersih)
        freq_csv = os.path.join(scope_dir, f"frequency_{feat}.csv")
        freq_tables_clean[feat].to_csv(freq_csv, index=False, encoding="utf-8-sig")
        rel_freq = os.path.relpath(freq_csv, os.path.join(app.config["UPLOAD_FOLDER"], "exports"))
        links["freq"].append((feat, url_for("download_export", subpath=rel_freq.replace("\\", "/"))))

        # Likelihood (reset_index agar header kiri = nama fitur)
        like_df_raw = like_tables_raw[feat]
        feat_name = like_df_raw.index.name or feat
        like_clean = like_df_raw.reset_index().rename(columns={feat_name: feat})
        like_csv = os.path.join(scope_dir, f"likelihood_{feat}.csv")
        like_clean.to_csv(like_csv, index=False, encoding="utf-8-sig")
        rel_like = os.path.relpath(like_csv, os.path.join(app.config["UPLOAD_FOLDER"], "exports"))
        links["like"].append((feat, url_for("download_export", subpath=rel_like.replace("\\", "/"))))
    return links

def build_split_links(base_dir, x_train, y_train, x_test, y_test, y_pred, target_col: str):
    split_dir = os.path.join(base_dir, "splits")
    ensure_dir(split_dir)

    y_train_df = y_train.to_frame(name=target_col)
    y_test_df  = y_test.to_frame(name=target_col)
    y_pred_df  = pd.DataFrame({"Predicted": y_pred})

    files = {
        "x_train": ("x_train.csv", x_train, False),
        "y_train": ("y_train.csv", y_train_df, False),
        "x_test":  ("x_test.csv",  x_test,  False),
        "y_test":  ("y_test.csv",  y_test_df,  False),
        "y_pred":  ("y_pred.csv",  y_pred_df,  False),
    }
    links = {}
    for key, (fname, df, use_index) in files.items():
        fpath = os.path.join(split_dir, fname)
        save_table(df, fpath, index=use_index)
        rel = os.path.relpath(fpath, os.path.join(app.config["UPLOAD_FOLDER"], "exports"))
        links[key] = url_for("download_export", subpath=rel.replace("\\", "/"))
    return links

# ================= Routes =================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            return render_template("index.html", error="Tidak ada file yang diupload.")
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        return redirect(url_for("analyze", filename=file.filename))
    return render_template("index.html")

@app.route("/exports/<path:subpath>")
def download_export(subpath):
    export_root = os.path.join(app.config["UPLOAD_FOLDER"], "exports")
    abs_path = os.path.join(export_root, subpath)
    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route("/analyze/<filename>")
def analyze(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(file_path)

    if df.isnull().any().any():
        return render_template("result.html", error="Dataset mengandung nilai kosong.")

    # Pisahkan fitur & target
    target_col = df.columns[-1]
    features = df.columns[:-1]
    X_full = df[features].astype(str)
    y_full = df[target_col].astype(str)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # =============== TRAIN ===============
    class_counts_train = y_train.value_counts()
    priors_train = class_counts_train / class_counts_train.sum()

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    train_df.columns = list(features) + [target_col]

    # FREQUENCY (TRAIN) -> kolom 1 = nama fitur
    freq_tables_train = {}
    for col in features:
        freq_raw = pd.crosstab(
            train_df[col], train_df[target_col],
            margins=True, margins_name="Grand Total"
        )
        freq_raw.columns.name = None
        freq_clean = freq_raw.reset_index()
        freq_clean.rename(columns={freq_clean.columns[0]: col}, inplace=True)
        freq_tables_train[col] = freq_clean

    # LIKELIHOOD (TRAIN)
    likelihoods_train_raw = {}   # disimpan mentah (index nilai + All)
    likelihoods_train_html = {}  # untuk ditampilkan (header kiri = nama fitur)
    for col in features:
        freq = pd.crosstab(train_df[col], train_df[target_col])
        like = freq.div(class_counts_train, axis=1)
        like = pd.concat([like, pd.DataFrame([priors_train], index=["All"])])
        like.columns.name = None
        like.index.name = col

        # versi untuk UI: reset_index supaya pojok kiri = nama fitur
        like_clean = like.reset_index().rename(columns={col: col})
        likelihoods_train_raw[col] = like
        likelihoods_train_html[col] = like_clean

    prior_table_train_df = pd.DataFrame({
        "Class": class_counts_train.index,
        "Count": class_counts_train.values,
        "Prior": priors_train.values
    })
    prior_table_train_html = prior_table_train_df.round(6).to_html(
        classes="table table-bordered table-sm", index=False, border=0
    )

    # Prediktor (TRAIN)
    def predict_row(row_dict):
        posteriors = {}
        for cls in class_counts_train.index:
            prob = float(priors_train[cls])
            for col, val in row_dict.items():
                val = str(val).strip()
                like_tab = likelihoods_train_raw[col]
                if val in like_tab.index:
                    prob *= float(like_tab.loc[val, cls])
                else:
                    prob *= 1e-9
            posteriors[cls] = prob
        total = sum(posteriors.values())
        if total > 0:
            for cls in posteriors:
                posteriors[cls] /= total
        return max(posteriors, key=posteriors.get)

    # Testing
    y_pred = [predict_row(X_test.iloc[i].to_dict()) for i in range(len(X_test))]

    accuracy = round((y_test.values == pd.Series(y_pred).values).mean() * 100, 2)
    report = classification_report(y_test, y_pred, output_dict=False)

    # Confusion matrix
    labels = sorted(y_full.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.rcParams["font.family"] = [
        "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI",
        "Roboto", "Arial", "Noto Sans", "Helvetica Neue", "sans-serif"
    ]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # =============== FULL (opsional) ===============
    class_counts_full = y_full.value_counts()
    priors_full = class_counts_full / class_counts_full.sum()

    freq_tables_full = {}
    for col in features:
        freq_raw = pd.crosstab(X_full[col], y_full, margins=True, margins_name="Grand Total")
        freq_raw.columns.name = None
        freq_clean = freq_raw.reset_index()
        freq_clean.rename(columns={freq_clean.columns[0]: col}, inplace=True)
        freq_tables_full[col] = freq_clean

    likelihoods_full_raw = {}
    likelihoods_full_html = {}
    for col in features:
        f = pd.crosstab(X_full[col], y_full)
        like = f.div(class_counts_full, axis=1)
        like = pd.concat([like, pd.DataFrame([priors_full], index=["All"])])
        like.columns.name = None
        like.index.name = col

        like_clean = like.reset_index().rename(columns={col: col})
        likelihoods_full_raw[col] = like
        likelihoods_full_html[col] = like_clean

    prior_table_full_df = pd.DataFrame({
        "Class": class_counts_full.index,
        "Count": class_counts_full.values,
        "Prior": priors_full.values
    })
    prior_table_full_html = prior_table_full_df.round(6).to_html(
        classes="table table-bordered table-sm", index=False, border=0
    )

    # =============== EXPORTS ===============
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    export_base = os.path.join(app.config["UPLOAD_FOLDER"], "exports", dataset_name)
    ensure_dir(export_base)

    train_links = build_export_links(
        export_base, "train", features, freq_tables_train, likelihoods_train_raw, prior_table_train_df
    )
    full_links = build_export_links(
        export_base, "full", features, freq_tables_full, likelihoods_full_raw, prior_table_full_df
    )
    split_links = build_split_links(export_base, X_train, y_train, X_test, y_test, y_pred, target_col)

    # Simpan state untuk prediksi manual
    global current_state
    current_state = {"features": list(features), "predict_row": predict_row}

    # Render
    return render_template(
        "result.html",
        filename=filename,
        head=df.to_html(classes="table table-striped", index=False),

        # Split view + unduh
        x_train=X_train.to_html(classes="table table-bordered table-sm", index=False),
        y_train=y_train.to_frame(name=target_col).to_html(classes="table table-bordered table-sm", index=False),
        x_test=X_test.to_html(classes="table table-bordered table-sm", index=False),
        y_test=y_test.to_frame(name=target_col).to_html(classes="table table-bordered table-sm", index=False),
        y_pred=pd.DataFrame({"Predicted": y_pred}).to_html(classes="table table-bordered table-sm", index=False),

        # TRAIN
        prior_table_train_html=prior_table_train_html,
        freq_tables_train={c: t.to_html(classes="table table-bordered table-sm", index=False, border=0)
                           for c, t in freq_tables_train.items()},
        likelihoods_train={c: t.round(3).to_html(classes="table table-bordered table-sm", index=False, border=0)
                           for c, t in likelihoods_train_html.items()},
        train_size=len(X_train),

        # FULL
        prior_table_full_html=prior_table_full_html,
        freq_tables_full={c: t.to_html(classes="table table-bordered table-sm", index=False, border=0)
                          for c, t in freq_tables_full.items()},
        likelihoods_full={c: t.round(3).to_html(classes="table table-bordered table-sm", index=False, border=0)
                          for c, t in likelihoods_full_html.items()},
        full_size=len(X_full),

        # links
        download_train=train_links,
        download_full=full_links,
        download_split=split_links,

        accuracy=accuracy,
        report=report,
        cm_base64=cm_base64
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    global current_state
    if not current_state:
        return redirect(url_for("index"))
    features = current_state["features"]
    if request.method == "POST":
        input_data = {f: request.form[f] for f in features}
        pred = current_state["predict_row"](input_data)
        return render_template("predict.html", features=features, result=pred)
    return render_template("predict.html", features=features, result=None)

if __name__ == "__main__":
    app.run(debug=True)
