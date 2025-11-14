import os
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.cm as cm
import threading
import logging
from logging.handlers import TimedRotatingFileHandler

# Ensure logs directory exists and configure a dedicated templates/access logger
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('rfm_app')
logger.setLevel(logging.INFO)
log_file = os.path.join(LOG_DIR, 'templates_access.log')
handler = TimedRotatingFileHandler(log_file, when='midnight', backupCount=7, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

app = Flask(__name__)

rfm_data = None
original_df = None
cached_df = None  # Cache raw data
cached_rfm = None  # Cache RFM data
cached_rfm_signature = None
cached_pie = None
cached_bar = None
cached_summary = None
cache_lock = threading.Lock()

def load_and_prepare_data():
    """Load and prepare the retail data (cached)"""
    global original_df, cached_df

    if cached_df is not None:
        print("✅ Using cached data")
        return cached_df

    # Try multiple paths in order of preference
    possible_paths = [
        'data/Online Retail.xlsx',
        'data/Online Retail.csv',
        'notebook/Online Retail.xlsx',
        'OnlineRetail.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                if path.lower().endswith('.xlsx'):
                    df = pd.read_excel(path, engine='openpyxl')
                else:
                    df = pd.read_csv(path)
                print(f"✅ Loaded dataset from: {path}")
                break
            except Exception as ex:
                print(f"⚠️ Failed to load from {path}: {ex}")
                continue
    
    if df is None:
        raise FileNotFoundError(f"Dataset not found. Tried: {possible_paths}")
    
    original_df = df.copy()

    df = df.dropna(subset=['CustomerID'])
    df = df.drop_duplicates()

    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    cached_df = df  # Cache the result
    return df

def create_rfm_analysis(df):
    """Create RFM analysis from transaction data (optimized)"""
    global cached_rfm
    
    if cached_rfm is not None:
        print("✅ Using cached RFM")
        return cached_rfm.copy()
    
    import datetime as dt

    ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',                              
        'TotalAmount': 'sum'                                 
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    cached_rfm = rfm  # Cache the result
    # Update signature and kick off background precompute of charts/summary
    try:
        sig = (len(rfm), float(rfm['Recency'].sum()), float(rfm['Frequency'].sum()), float(rfm['Monetary'].sum()))
        global cached_rfm_signature
        cached_rfm_signature = sig
        # Precompute charts in background to keep UI snappy
        threading.Thread(target=_precompute_artifacts, args=(rfm.copy(), sig), daemon=True).start()
    except Exception:
        logger.exception('Failed to start background precompute')
    return rfm


def _precompute_artifacts(rfm, signature):
    """Precompute segmentation, charts and summary for a given RFM snapshot.

    Stores results in module-level cached_pie/cached_bar/cached_summary if the
    signature still matches (avoids overwriting newer data).
    """
    try:
        seg = perform_segmentation(rfm, n_clusters=7)
        pie = generate_pie_chart(seg)
        bar = generate_bar_chart(seg)
        segment_summary = seg['Segment'].value_counts().reindex(
            ['Champions', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Lost', 'Others'],
            fill_value=0
        )
        summary_data = [{'segment': str(name), 'size': int(count)} for name, count in segment_summary.items()]

        # store atomically
        with cache_lock:
            global cached_rfm_signature, cached_pie, cached_bar, cached_summary
            if cached_rfm_signature == signature:
                cached_pie = pie
                cached_bar = bar
                cached_summary = summary_data
                logger.info('Background precompute: artifacts cached')
            else:
                logger.info('Background precompute: signature mismatch, discarding results')
        # Also persist to disk so images survive restarts and are immediately
        # available to the dashboard even if in-memory caches are lost.
        try:
            artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            pie_path = os.path.join(artifacts_dir, 'pie.png')
            bar_path = os.path.join(artifacts_dir, 'bar.png')
            with open(pie_path, 'wb') as f:
                f.write(base64.b64decode(pie))
            with open(bar_path, 'wb') as f:
                f.write(base64.b64decode(bar))
            logger.info('Background precompute: artifacts persisted to disk')
        except Exception:
            logger.exception('Failed to persist precomputed artifacts to disk')
    except Exception:
        logger.exception('Error during background precompute of artifacts')

def perform_segmentation(rfm, n_clusters=7):
    """Vectorized RFM segmentation using quartiles and deterministic rules.

    This replaces Python-level row-wise apply with vectorized operations for speed.
    The function still accepts `n_clusters` for API compatibility but segmentation
    uses fixed business rules and quartiles to produce the seven named segments.
    """
    df = rfm.copy()

    # Ensure n_clusters is in expected range (kept for compatibility)
    try:
        n_clusters = max(2, min(int(n_clusters), 7))
    except Exception:
        n_clusters = 7

    # 1) Create RFM scores using quartiles
    # Recency: lower is better -> reverse labels
    df['R_Score'] = pd.qcut(df['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype('int8')

    # Frequency: rank first to stabilize ties, then quartile
    df['F_Rank'] = df['Frequency'].rank(method='first')
    df['F_Score'] = pd.qcut(df['F_Rank'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype('int8')

    # Monetary
    df['M_Score'] = pd.qcut(df['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype('int8')

    # Combined RFM score (optional, useful for ordering or tie-breaking)
    df['RFM_Score'] = (df['R_Score'].astype('int16') + df['F_Score'].astype('int16') + df['M_Score'].astype('int16'))

    # 2) Vectorized rule assignment using numpy.select
    import numpy as np

    conds = [
        (df['R_Score'] == 4) & (df['F_Score'] == 4) & (df['M_Score'] == 4),
        (df['F_Score'] >= 3) & (df['M_Score'] >= 3) & (df['R_Score'] >= 2),
        (df['R_Score'] >= 3) & (df['F_Score'] <= 2),
        (df['R_Score'] >= 3) & (df['M_Score'] <= 2),
        (df['R_Score'] == 2) & (df['F_Score'] <= 2),
        (df['R_Score'] == 1) & (df['F_Score'] == 1)
    ]

    choices = [
        'Champions',
        'Loyal Customers',
        'Potential Loyalists',
        'Promising',
        'At Risk',
        'Lost'
    ]

    df['Segment'] = np.select(conds, choices, default='Others')

    # Cleanup temporary columns we don't need in downstream templates
    df = df.drop(columns=['F_Rank'], errors='ignore')

    return df

def generate_pie_chart(rfm):
    """Generate pie chart for customer segments (optimized)"""
    segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Lost', 'Others']
    cluster_counts = rfm['Segment'].value_counts().reindex(segment_order, fill_value=0)
    
    # Filter out segments with 0 count
    cluster_counts = cluster_counts[cluster_counts > 0]
    n_segments = len(cluster_counts)

    if n_segments <= 3:
        colors = ['#5B9BD5', '#70AD47', '#FFC000'][:n_segments]
    else:
        colors = cm.tab20(range(n_segments))

    plt.figure(figsize=(7, 5), dpi=60)  # Reduced size and DPI for faster render
    plt.pie(
        cluster_counts.values,
        labels=[str(name) for name in cluster_counts.index],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9}
    )
    plt.title('Customer Segments Distribution', fontsize=12, pad=15)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=60)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

def generate_bar_chart(rfm):
    """Generate bar chart for segment characteristics (optimized)"""
    segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Lost', 'Others']
    cluster_sizes = rfm['Segment'].value_counts().reindex(segment_order, fill_value=0)
    
    # Filter out segments with 0 count
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    n_segments = len(cluster_sizes)

    if n_segments <= 3:
        colors = ['#5B9BD5', '#70AD47', '#FFC000'][:n_segments]
    else:
        colors = cm.tab20(range(n_segments))

    fig_width = max(7, n_segments * 0.9)
    plt.figure(figsize=(fig_width, 3.5), dpi=60)  # Reduced size and DPI

    x_positions = range(len(cluster_sizes))
    bars = plt.bar(x_positions, cluster_sizes.values, color=colors)

    for bar, value in zip(bars, cluster_sizes.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontsize=9)

    plt.title('Customers per Segment', fontsize=12)
    plt.xlabel('Segments', fontsize=10)
    plt.ylabel('Count', fontsize=10)

    plt.xticks(x_positions, [str(name) for name in cluster_sizes.index], rotation=45, ha='right', fontsize=9)
    plt.grid(axis='y', alpha=0.2, linestyle='--')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=60)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

@app.route('/')
def home():
    """Home page"""
    logger.info('Serving template: home.html')
    return render_template('home.html')

@app.route('/dataset')
def dataset():
    """Dataset page"""
    global original_df
    # If we already have the full prepared dataset cached, use it (fast)
    if cached_df is not None:
        df_preview = cached_df.head(100).copy()
        logger.info('Serving dataset preview from cache')
    else:
        # Try to serve a lightweight preview quickly by reading a small sample.
        # Prefer small CSV previews if present to avoid loading the full Excel file on first request.
        preview_paths = [
            'artifacts/test.csv',
            'artifacts/raw_data.csv',
            'data/Online Retail.csv',
            'data/Online Retail.xlsx'
        ]
        df_preview = None
        for p in preview_paths:
            try:
                if os.path.exists(p):
                    if p.lower().endswith('.xlsx'):
                        # read only first 100 rows from excel (fast)
                        df_preview = pd.read_excel(p, engine='openpyxl', nrows=100)
                    else:
                        df_preview = pd.read_csv(p, nrows=100)
                    logger.info('Serving dataset preview from %s', p)
                    break
            except Exception as e:
                logger.warning('Preview load failed for %s: %s', p, str(e))
                continue

        # If still None, fall back to calling the full loader (but do it asynchronously)
        if df_preview is None:
            # spawn background thread to warm the cache while returning a minimal message
            def _warm_cache():
                try:
                    load_and_prepare_data()
                    logger.info('Background cache warm complete')
                except Exception:
                    logger.exception('Background cache warm failed')

            threading.Thread(target=_warm_cache, daemon=True).start()
            # minimal placeholder until cache is ready
            df_preview = pd.DataFrame(columns=['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country'])

    # Only keep columns the template expects and convert InvoiceDate to string for safe rendering
    cols = ['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']
    df_preview = df_preview.loc[:, [c for c in cols if c in df_preview.columns]]
    if 'InvoiceDate' in df_preview.columns:
        df_preview['InvoiceDate'] = df_preview['InvoiceDate'].astype(str)

    data_preview = df_preview.head(100).to_dict('records')
    columns = df_preview.columns.tolist()

    logger.info('Serving template: dataset.html (preview rows=%d)', len(data_preview))
    return render_template('dataset.html', data=data_preview, columns=columns)

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    logger.info('Serving template: dashboard.html')
    return render_template('dashboard.html')

@app.route('/generate_segments', methods=['POST'])
def generate_segments():
    """Generate customer segments based on number of clusters"""
    global rfm_data, cached_pie, cached_bar, cached_summary, cached_rfm_signature, cached_rfm

    try:
        # Server-side: force number of segments to 7 (fixed)
        n_clusters = 7
        print(f"Loading data for {n_clusters} (fixed) segments...")
        logger.info('Request: /generate_segments - start (n_clusters=%d)', n_clusters)

        # Ensure RFM is available (loads and caches if needed)
        if cached_rfm is None:
            df = load_and_prepare_data()
            rfm_data = create_rfm_analysis(df)
        else:
            rfm_data = cached_rfm

        # Fast path: if artifacts already precomputed and signature matches, return them
        # We'll wait briefly for any background precompute to finish (helps avoid
        # race where the background thread is still running) and also fall back to
        # on-disk artifacts if in-memory caches are empty.
        import time
        wait_until = time.time() + 2.0  # up to 2 seconds
        while time.time() < wait_until:
            with cache_lock:
                cp = globals().get('cached_pie')
                cb = globals().get('cached_bar')
                cs = globals().get('cached_summary')
                if cached_rfm_signature is not None and cp is not None and cb is not None and cs is not None:
                    logger.info('Returning cached artifacts for /generate_segments (in-memory)')
                    try:
                        logger.info('Cached pie size: %d, bar size: %d', len(cp), len(cb))
                    except Exception:
                        logger.info('Cached artifacts present but could not get sizes')
                    return jsonify({
                        'success': True,
                        'pie_chart': cp,
                        'bar_chart': cb,
                        'summary': cs,
                        'n_clusters': 7
                    })
            time.sleep(0.15)

        # If in-memory cache wasn't ready, try on-disk artifacts (persisted by precompute)
        try:
            artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
            pie_path = os.path.join(artifacts_dir, 'pie.png')
            bar_path = os.path.join(artifacts_dir, 'bar.png')
            if os.path.exists(pie_path) and os.path.exists(bar_path):
                with open(pie_path, 'rb') as f:
                    pie_bytes = f.read()
                with open(bar_path, 'rb') as f:
                    bar_bytes = f.read()
                cp = base64.b64encode(pie_bytes).decode()
                cb = base64.b64encode(bar_bytes).decode()
                # Rebuild a minimal summary from the segmentation (if available)
                cs = globals().get('cached_summary')
                if cs is None and cached_rfm is not None:
                    tmp_seg = perform_segmentation(cached_rfm, n_clusters=n_clusters)
                    segment_summary = tmp_seg['Segment'].value_counts().reindex(
                        ['Champions', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Lost', 'Others'],
                        fill_value=0
                    )
                    cs = [{'segment': str(name), 'size': int(count)} for name, count in segment_summary.items()]
                if cp and cb and cs:
                    logger.info('Returning cached artifacts for /generate_segments (from disk)')
                    return jsonify({
                        'success': True,
                        'pie_chart': cp,
                        'bar_chart': cb,
                        'summary': cs,
                        'n_clusters': 7
                    })
        except Exception:
            logger.exception('Failed to load persisted artifacts from disk')

        # Fallback: compute synchronously if no cached artifacts yet
        logger.info('No cached artifacts found; computing synchronously')
        seg_df = perform_segmentation(rfm_data, n_clusters=n_clusters)
        pie_chart = generate_pie_chart(seg_df)
        bar_chart = generate_bar_chart(seg_df)
        segment_summary = seg_df['Segment'].value_counts().reindex(
            ['Champions', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Lost', 'Others'],
            fill_value=0
        )
        summary_data = [{'segment': str(name), 'size': int(count)} for name, count in segment_summary.items()]

        # store computed artifacts for subsequent fast responses (set via globals to avoid scope issues)
        with cache_lock:
            globals()['cached_pie'] = pie_chart
            globals()['cached_bar'] = bar_chart
            globals()['cached_summary'] = summary_data

        print(f"✅ Success! {len(summary_data)} segments generated")
        logger.info('Success: generated %d segments (sync fallback)', len(summary_data))
        try:
            logger.info('Generated pie size: %d, bar size: %d', len(pie_chart), len(bar_chart))
        except Exception:
            logger.info('Could not determine generated artifact sizes')
        return jsonify({
            'success': True,
            'pie_chart': pie_chart,
            'bar_chart': bar_chart,
            'summary': summary_data,
            'n_clusters': 7
        })

    except Exception as e:
        print(f"❌ Error in /generate_segments: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Run without debug/reloader for stable behavior while testing
    # Use PORT env var for Render deployment, default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')
