import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fpdf import FPDF
import tempfile
import os

# ── Konfigurasi halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="Product Trend Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Product Trend Predictor")
st.markdown("Upload data Google Trends kamu → sistem otomatis prediksi tren produk!")
st.divider()

# ── Fungsi ──────────────────────────────────────────────────────
def baca_csv(file, nama_produk):
    df = pd.read_csv(file, skiprows=1)
    df.columns = ['tanggal', nama_produk]
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df[nama_produk] = pd.to_numeric(df[nama_produk], errors='coerce').astype(float)
    df = df.set_index('tanggal')
    return df

def bersihkan_outlier(series):
    series = series.copy().astype(float)
    threshold = series.quantile(0.9)
    mean_normal = series[series < threshold].mean()
    series[series > threshold] = mean_normal
    return series

def prediksi_tren(series, minggu=12):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    future_X = poly.transform(np.arange(len(series), len(series)+minggu).reshape(-1,1))
    future_pred = np.clip(model.predict(future_X), 0, 100)
    future_dates = pd.date_range(start=series.index[-1], periods=minggu+1, freq='W')[1:]
    return future_dates, future_pred

def buat_grafik(produk, info, minggu_prediksi):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(info['series_asli'].index, info['series_asli'].values,
            color='lightblue', alpha=0.7, label='Data Asli')
    ax.plot(info['series_clean'].index, info['series_clean'].values,
            color='blue', linewidth=1.5, label='Data Bersih')
    ax.plot(info['future_dates'], info['future_pred'],
            color='red', linewidth=2, linestyle='--',
            marker='o', markersize=3, label=f'Prediksi {minggu_prediksi} Minggu')
    ax.fill_between(info['future_dates'],
                    info['future_pred'] - 3, info['future_pred'] + 3,
                    alpha=0.2, color='red')
    ax.axvline(x=info['series_clean'].index[-1], color='gray', linestyle=':')
    ax.set_title(f"{produk.upper()} ({info['perubahan']:+.1f}%)")
    ax.set_ylabel('Minat (0-100)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def buat_pdf(hasil, minggu_prediksi, ranking):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 12, 'Product Trend Predictor', ln=True, align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, f'Laporan Prediksi Tren - {minggu_prediksi} Minggu ke Depan', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
    pdf.ln(5)

    # Ranking
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Ranking Produk', ln=True)
    pdf.set_font('Helvetica', '', 11)
    for i, (produk, info) in enumerate(ranking, 1):
        status = 'RECOMMENDED' if info['perubahan'] > 0 else 'HINDARI'
        pdf.cell(0, 8, f"{i}. {produk.upper()}  {info['perubahan']:+.1f}%  [{status}]", ln=True)
    pdf.ln(5)

    # Grafik per produk
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Grafik Tren & Prediksi', ln=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for produk, info in hasil.items():
            fig = buat_grafik(produk, info, minggu_prediksi)
            img_path = os.path.join(tmpdir, f'{produk}.png')
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, f'{produk.upper()} ({info["perubahan"]:+.1f}%)', ln=True)
            pdf.image(img_path, w=180)
            pdf.ln(3)

            # Tabel prediksi
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(50, 7, 'Tanggal', border=1, fill=True)
            pdf.cell(50, 7, 'Prediksi Minat', border=1, fill=True)
            pdf.cell(50, 7, 'Status', border=1, fill=True, ln=True)

            pdf.set_font('Helvetica', '', 10)
            for i, (date, pred) in enumerate(zip(info['future_dates'], info['future_pred'])):
                status = 'Naik' if i == 0 or pred > info['future_pred'][i-1] else 'Turun'
                pdf.cell(50, 6, date.strftime('%Y-%m-%d'), border=1)
                pdf.cell(50, 6, f'{pred:.1f}', border=1)
                pdf.cell(50, 6, status, border=1, ln=True)
            pdf.ln(8)

    return bytes(pdf.output())

# ── Upload Section ───────────────────────────────────────────────
st.subheader("📂 Upload Data Google Trends")
st.info("Download CSV dari trends.google.com lalu upload di sini. Bisa upload lebih dari 1 file!")

uploaded_files = st.file_uploader(
    "Pilih file CSV Google Trends",
    type=['csv'],
    accept_multiple_files=True
)

if uploaded_files:
    st.divider()

    st.subheader("🏷️ Beri nama setiap produk")
    nama_produk = {}
    cols = st.columns(len(uploaded_files))
    for i, file in enumerate(uploaded_files):
        with cols[i]:
            nama = st.text_input(
                f"Nama untuk '{file.name}'",
                value=file.name.replace('.csv','').replace('Data_TrenProduk','').lower(),
                key=f"nama_{i}"
            )
            nama_produk[file.name] = nama

    minggu_prediksi = st.slider("📅 Prediksi berapa minggu ke depan?", 4, 24, 12)

    if st.button("🚀 Analisis Sekarang!", type="primary", use_container_width=True):

        with st.spinner("Sedang menganalisis data..."):
            semua_df = {}
            for file in uploaded_files:
                nama = nama_produk[file.name]
                df_temp = baca_csv(file, nama)
                semua_df[nama] = df_temp

            df_all = pd.concat(semua_df.values(), axis=1)

            hasil = {}
            for produk in df_all.columns:
                series_clean = bersihkan_outlier(df_all[produk])
                future_dates, future_pred = prediksi_tren(series_clean, minggu_prediksi)
                perubahan = ((future_pred[-1] - future_pred[0]) / max(future_pred[0], 1)) * 100
                hasil[produk] = {
                    'series_asli': df_all[produk],
                    'series_clean': series_clean,
                    'future_dates': future_dates,
                    'future_pred': future_pred,
                    'perubahan': perubahan
                }

        st.success("✅ Analisis selesai!")
        st.divider()

        # Ranking
        st.subheader("🏆 Ranking Produk")
        ranking = sorted(hasil.items(), key=lambda x: x[1]['perubahan'], reverse=True)

        cols_rank = st.columns(len(ranking))
        for i, (produk, info) in enumerate(ranking):
            with cols_rank[i]:
                warna = "normal" if info['perubahan'] > 0 else "inverse"
                st.metric(
                    label=produk.upper(),
                    value=f"{info['perubahan']:+.1f}%",
                    delta="🔥 RECOMMENDED" if info['perubahan'] > 0 else "⚠️ HINDARI",
                    delta_color=warna
                )

        st.divider()

        # Grafik
        st.subheader("📊 Grafik Tren & Prediksi")
        cols_chart = st.columns(2)
        for i, (produk, info) in enumerate(hasil.items()):
            with cols_chart[i % 2]:
                fig = buat_grafik(produk, info, minggu_prediksi)
                st.pyplot(fig)
                plt.close()

        st.divider()

        # Tabel
        st.subheader("📋 Detail Prediksi Mingguan")
        tabs = st.tabs([p.upper() for p in hasil.keys()])
        for tab, (produk, info) in zip(tabs, hasil.items()):
            with tab:
                df_pred = pd.DataFrame({
                    'Tanggal': info['future_dates'].strftime('%Y-%m-%d'),
                    'Prediksi Minat': info['future_pred'].round(1),
                    'Status': ['🔥 Naik' if i == 0 or info['future_pred'][i] > info['future_pred'][i-1]
                               else '⬇️ Turun' for i in range(len(info['future_pred']))]
                })
                st.dataframe(df_pred, use_container_width=True)

        st.divider()

        # Download
        st.subheader("⬇️ Download Hasil")
        col1, col2 = st.columns(2)

        with col1:
            df_download = pd.DataFrame()
            for produk, info in hasil.items():
                df_temp = pd.DataFrame({
                    'tanggal': info['future_dates'].strftime('%Y-%m-%d'),
                    f'{produk}_prediksi': info['future_pred'].round(1)
                }).set_index('tanggal')
                df_download = pd.concat([df_download, df_temp], axis=1)

            csv_result = df_download.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_result,
                file_name='hasil_prediksi_tren.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            with st.spinner("Menyiapkan PDF..."):
                pdf_bytes = buat_pdf(hasil, minggu_prediksi, ranking)
            st.download_button(
                label="📄 Download PDF (dengan grafik)",
                data=pdf_bytes,
                file_name='laporan_prediksi_tren.pdf',
                mime='application/pdf',
                use_container_width=True
            )

else:
    st.markdown("""
    ### 📌 Cara Pakai:
    1. Buka **trends.google.com**
    2. Cari keyword produk yang ingin dianalisis
    3. Set lokasi **Indonesia** dan periode **12 bulan**
    4. Klik tombol download CSV
    5. Upload file CSV di atas
    6. Klik **Analisis Sekarang!**
    """)