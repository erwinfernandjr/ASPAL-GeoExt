import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import random
import math
from shapely.geometry import Point, LineString
import rasterio
from rasterstats import zonal_stats
import io

# Import untuk ReportLab (PDF)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Import untuk Evaluasi Akurasi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================================
# 1. KONFIGURASI HALAMAN UTAMA
# =========================================
st.set_page_config(page_title="ASPAL GeoExt", page_icon="📐", layout="wide")

# ==========================================
# 2. FUNGSI SPASIAL & BANTUAN
# ==========================================
def read_zip_shapefile(uploaded_file, tmpdir):
    zip_path = os.path.join(tmpdir, uploaded_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    extract_dir = os.path.join(tmpdir, uploaded_file.name.replace('.zip', ''))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".shp"):
                return gpd.read_file(os.path.join(root, file))
    return None

def hitung_geometri_dasar(gdf):
    gdf = gdf.copy()
    panjang, lebar, diameter, luas = [], [], [], []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            panjang.append(0); lebar.append(0); diameter.append(0); luas.append(0)
            continue
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [LineString([coords[i], coords[i+1]]).length for i in range(4)]
        unique_edges = sorted(list(set([round(e, 5) for e in edges])))
        p = max(unique_edges) if unique_edges else 0
        l = min(unique_edges) if unique_edges else 0
        d = (p + l) / 2 if len(unique_edges) >= 2 else p 
        panjang.append(p); lebar.append(l); diameter.append(d); luas.append(geom.area)
    gdf["Panjang_m"] = np.round(panjang, 3); gdf["Lebar_m"] = np.round(lebar, 3)
    gdf["Diameter_m"] = np.round(diameter, 3); gdf["Luas_m2"] = np.round(luas, 3)
    return gdf

def hitung_kedalaman(gdf, dsm_path, buffer_distance=0.3):
    gdf = gdf.copy()
    with rasterio.open(dsm_path) as DSM:
        dsm_crs = DSM.crs
        nodata_val = DSM.nodata
    buffer_outer = gdf.geometry.buffer(buffer_distance)
    ring_geom = buffer_outer.difference(gdf.geometry)
    hole_geom_dsm = gdf.geometry.to_crs(dsm_crs)
    ring_geom_dsm = ring_geom.to_crs(dsm_crs)
    stats_hole = zonal_stats(hole_geom_dsm, dsm_path, stats=["median"], nodata=nodata_val, all_touched=True)
    stats_ring = zonal_stats(ring_geom_dsm, dsm_path, stats=["median"], nodata=nodata_val, all_touched=True)
    kedalaman = []
    for i in range(len(gdf)):
        z_hole = stats_hole[i]["median"]
        z_ref = stats_ring[i]["median"]
        depth = max(0, (z_ref - z_hole)) if z_hole is not None and z_ref is not None else 0
        kedalaman.append(depth)
    gdf["Kedalaman_m"] = np.round(kedalaman, 4)
    return gdf

def generate_pdf_report(df_rekap, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>LAPORAN EKSTRAKSI GEOMETRI</b>", styles['Title']))
    elements.append(Paragraph("<b>ASPAL GeoExt (Geometry Extractor)</b>", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))
    tabel_data = [df_rekap.columns.tolist()] + df_rekap.values.tolist()
    t = Table(tabel_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc"))
    ]))
    elements.append(t); doc.build(elements)

def generate_pdf_eval_report(df_report, acc, cm_image_path, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<b>LAPORAN EVALUASI AKURASI MODEL</b>", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"<b>Overall Accuracy:</b> {acc:.2%}", styles['Heading3']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Masukkan Gambar Confusion Matrix
    if os.path.exists(cm_image_path):
        elements.append(RLImage(cm_image_path, width=6*inch, height=4.5*inch))
        elements.append(Spacer(1, 0.3 * inch))
    
    elements.append(Paragraph("<b>Detail Metrik Klasifikasi</b>", styles['Heading3']))
    
    # Siapkan tabel metrik
    df_report = df_report.round(2)
    df_report.reset_index(inplace=True)
    df_report.rename(columns={'index': 'Kelas'}, inplace=True)
    
    tabel_data = [df_report.columns.tolist()] + df_report.values.tolist()
    t = Table(tabel_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc"))
    ]))
    
    elements.append(t)
    doc.build(elements)

def get_random_points_gdf(gdf, target_n, is_background=False):
    points = []
    if gdf.empty or target_n <= 0: return []
    geometries = gdf.geometry.tolist()
    
    if is_background:
        attempts = 0
        max_attempts = target_n * 200 
        while len(points) < target_n and attempts < max_attempts:
            poly = random.choice(geometries)
            minx, miny, maxx, maxy = poly.bounds
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p): points.append(p)
            attempts += 1
    else:
        actual_n = min(target_n, len(geometries))
        selected_polys = random.sample(geometries, actual_n)
        for poly in selected_polys:
            minx, miny, maxx, maxy = poly.bounds
            poly_attempts = 0
            while poly_attempts < 100: 
                p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if poly.contains(p):
                    points.append(p)
                    break 
                poly_attempts += 1
    return points

def get_random_points_raster(raster_path, n_points, epsg_code):
    points = []
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_crs = src.crs
        nodata = src.nodata
        attempts = 0
        max_attempts = n_points * 200
        while len(points) < n_points and attempts < max_attempts:
            x = random.uniform(bounds.left, bounds.right)
            y = random.uniform(bounds.bottom, bounds.top)
            try:
                val = next(src.sample([(x, y)]))
                if (nodata is None or val[0] != nodata) and np.any(val > 0):
                    points.append(Point(x, y))
            except: pass
            attempts += 1
            
    if points:
        gdf_temp = gpd.GeoDataFrame(geometry=points, crs=raster_crs)
        return gdf_temp.to_crs(epsg=epsg_code).geometry.tolist()
    return []

def assign_deteksi_class(class_name):
    mapping = {
        "Alligator Crack": 1, "Edge Crack": 2, "Longitudinal Crack": 3,
        "Patching": 4, "Potholes": 5, "Rutting": 6, "Non-Distress": 7
    }
    return mapping.get(class_name, 0)

# ==========================================
# 3. SIDEBAR & NAVIGASI
# ==========================================
st.markdown("<h1 style='text-align: center; color: #4da6ff;'>📐 ASPAL GeoExt</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #cbd5e1;'>Sistem Ekstraksi Geometri & Evaluasi Berbasis GIS</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("🧭 Navigasi Modul")
    menu = st.radio("Pilih Modul:", [
        "📏 Ekstraksi Geometri Kerusakan",
        "🎯 Random Sampling (Ground Truth)",
        "📊 Evaluasi Akurasi (Confusion Matrix)"
    ])
    
    st.divider()
    st.header("⚙️ Pengaturan Global")
    epsg_code = st.number_input("Kode EPSG Proyeksi (Meters)", value=32749, step=1)
    
    if st.button("🔄 Reset Modul", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ==========================================
# MODUL 1: EKSTRAKSI GEOMETRI
# ==========================================
if menu == "📏 Ekstraksi Geometri Kerusakan":
    st.subheader("Modul Ekstraksi Dimensi Kerusakan")
    
    if 'geoext_selesai' not in st.session_state: st.session_state.geoext_selesai = False

    col_input1, col_input2 = st.columns([1.5, 1])
    with col_input1:
        st.markdown("**1. Input Shapefile Kerusakan (.zip)**")
        c1, c2 = st.columns(2)
        with c1:
            file_ac = st.file_uploader("Alligator Crack", type="zip", key="ac")
            file_ec = st.file_uploader("Edge Crack", type="zip", key="ec")
            file_lc = st.file_uploader("Longitudinal Crack", type="zip", key="lc")
        with c2:
            file_pt = st.file_uploader("Potholes", type="zip", key="pt")
            file_pa = st.file_uploader("Patching", type="zip", key="pa")
            file_rt = st.file_uploader("Rutting", type="zip", key="rt")

    with col_input2:
        st.markdown("**2. Input Data Kedalaman (DSM)**")
        dsm_mode = st.radio("Cara Input DSM:", ["Paste Link Google Drive", "Upload File .tif"])
        dsm_file = None; dsm_link = ""
        if dsm_mode == "Upload File .tif": dsm_file = st.file_uploader("Upload DSM (.tif)", type="tif")
        else: dsm_link = st.text_input("Paste Link Drive DSM (.tif)")

    if st.button("🚀 Ekstrak Geometri", type="primary", use_container_width=True):
        konfigurasi_input = {
            "Alligator Crack": {"file": file_ac, "kebutuhan_dsm": False},
            "Edge Crack": {"file": file_ec, "kebutuhan_dsm": False},
            "Longitudinal Crack": {"file": file_lc, "kebutuhan_dsm": False},
            "Potholes": {"file": file_pt, "kebutuhan_dsm": True},
            "Patching": {"file": file_pa, "kebutuhan_dsm": False},
            "Rutting": {"file": file_rt, "kebutuhan_dsm": True}
        }
        
        with st.spinner("Memproses perhitungan geometri..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    dsm_path = None
                    is_dsm_valid = (dsm_mode == "Upload File .tif" and dsm_file is not None) or (dsm_mode == "Paste Link Google Drive" and dsm_link != "")
                    if is_dsm_valid:
                        dsm_path = os.path.join(tmpdir, "dsm.tif")
                        if dsm_mode == "Upload File .tif":
                            with open(dsm_path, "wb") as f: f.write(dsm_file.getbuffer())
                        else:
                            import gdown, re
                            match = re.search(r"/d/([a-zA-Z0-9_-]+)", dsm_link)
                            if match: gdown.download(id=match.group(1), output=dsm_path, quiet=False)

                    hasil_gdf_list, rekap_data = [], []
                    for jk, config in konfigurasi_input.items():
                        file = config["file"]
                        butuh_dsm = config["kebutuhan_dsm"]
                        if file is not None:
                            gdf = read_zip_shapefile(file, tmpdir)
                            if gdf is not None and not gdf.empty:
                                if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
                                gdf = gdf.to_crs(epsg=epsg_code)
                                gdf = hitung_geometri_dasar(gdf)
                                gdf["Kedalaman_m"] = 0.0
                                if butuh_dsm and dsm_path: gdf = hitung_kedalaman(gdf, dsm_path)
                                gdf["Jenis_Kerusakan"] = jk
                                hasil_gdf_list.append(gdf)
                                rekap_data.append({
                                    "Jenis Kerusakan": jk, "Total Objek": len(gdf),
                                    "Rata P (m)": round(gdf["Panjang_m"].mean(), 3), "Rata L (m)": round(gdf["Lebar_m"].mean(), 3),
                                    "Rata D (m)": round(gdf["Diameter_m"].mean(), 3), "Total Luas (m²)": round(gdf["Luas_m2"].sum(), 3),
                                    "Rata Z (m)": round(gdf["Kedalaman_m"].mean(), 3) if butuh_dsm and dsm_path else "-"
                                })

                    if not hasil_gdf_list: st.error("❌ Tidak ada file Shapefile valid diunggah."); st.stop()

                    master_gdf = pd.concat(hasil_gdf_list, ignore_index=True)
                    df_rekap = pd.DataFrame(rekap_data)

                    gpkg_path = os.path.join(tmpdir, "GeoExt_Hasil.gpkg")
                    master_gdf.to_file(gpkg_path, driver="GPKG")
                    
                    excel_buffer = io.BytesIO()
                    df_detail = master_gdf.drop(columns=['geometry'])
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df_rekap.to_excel(writer, sheet_name='Rekapitulasi', index=False)
                        df_detail.to_excel(writer, sheet_name='Detail Ekstraksi', index=False)
                    
                    pdf_path = os.path.join(tmpdir, "GeoExt_Laporan.pdf")
                    generate_pdf_report(df_rekap, pdf_path)

                    st.session_state.df_rekap_geo = df_rekap
                    st.session_state.df_detail_geo = df_detail
                    with open(gpkg_path, "rb") as f: st.session_state.gpkg_bytes_geo = f.read()
                    st.session_state.excel_bytes_geo = excel_buffer.getvalue()
                    with open(pdf_path, "rb") as f: st.session_state.pdf_bytes_geo = f.read()
                    st.session_state.geoext_selesai = True

                except Exception as e: st.error(f"❌ Kesalahan: {e}"); st.session_state.geoext_selesai = False

    if st.session_state.get('geoext_selesai', False):
        st.success("✅ Ekstraksi selesai!")
        st.dataframe(st.session_state.df_rekap_geo, use_container_width=True, hide_index=True)
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1: st.download_button("📄 PDF", data=st.session_state.pdf_bytes_geo, file_name="GeoExt_Laporan.pdf", mime="application/pdf", use_container_width=True)
        with col_dl2: st.download_button("📊 Excel", data=st.session_state.excel_bytes_geo, file_name="GeoExt_Data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col_dl3: st.download_button("🗺️ GPKG", data=st.session_state.gpkg_bytes_geo, file_name="GeoExt_Peta.gpkg", mime="application/geopackage+sqlite3", use_container_width=True)

# ==========================================
# MODUL 2: RANDOM SAMPLING
# ==========================================
elif menu == "🎯 Random Sampling (Ground Truth)":
    st.subheader("Modul Generator Sampel Acak")
    
    if 'sampling_selesai' not in st.session_state: st.session_state.sampling_selesai = False

    st.markdown("### 🧮 Metode Penentuan Jumlah Sampel")
    metode_sampling = st.selectbox(
        "Pilih pendekatan kalkulasi sampel untuk kelas kerusakan:",
        ["Input Manual", "Rumus Slovin", "Aturan Roscoe", "Persentase Populasi"]
    )

    n_manual = 50; e_slovin = 0.05; p_persen = 0.10

    if metode_sampling == "Input Manual":
        st.info("💡 **Input Manual:** Anda menetapkan target angka pasti. Jika poligon kerusakan lebih sedikit dari target, semua poligon akan diambil.")
        n_manual = st.number_input("Target Maksimal Sampel Titik per Kelas", min_value=1, value=50)
    elif metode_sampling == "Rumus Slovin":
        st.info("💡 **Rumus Slovin:** Ukuran sampel dihitung dinamis berdasarkan populasi ($N$) tiap jenis kerusakan. $n = N / (1 + N \cdot e^2)$")
        e_input = st.number_input("Batas Toleransi Error / Margin of Error (%)", min_value=1, max_value=50, value=5)
        e_slovin = e_input / 100.0
    elif metode_sampling == "Aturan Roscoe":
        st.info("💡 **Aturan Roscoe (1975):** Ukuran sampel minimum yang layak untuk penelitian adalah 30 per kategori/kelas.")
        n_manual = 30 
    elif metode_sampling == "Persentase Populasi":
        st.info("💡 **Persentase:** Mengambil sekian persen dari total populasi ($N$) tiap jenis kerusakan.")
        p_input = st.number_input("Persentase Sampel dari Populasi (%)", min_value=1, max_value=100, value=10)
        p_persen = p_input / 100.0

    st.divider()
    
    col_samp1, col_samp2 = st.columns([1.5, 1])
    with col_samp1:
        st.markdown("**1. Shapefile Target Kerusakan (.zip)**")
        c1, c2 = st.columns(2)
        with c1:
            file_ac_s = st.file_uploader("Alligator Crack", type="zip", key="ac_s")
            file_ec_s = st.file_uploader("Edge Crack", type="zip", key="ec_s")
            file_lc_s = st.file_uploader("Longitudinal Crack", type="zip", key="lc_s")
        with c2:
            file_pt_s = st.file_uploader("Potholes", type="zip", key="pt_s")
            file_pa_s = st.file_uploader("Patching", type="zip", key="pa_s")
            file_rt_s = st.file_uploader("Rutting", type="zip", key="rt_s")

    with col_samp2:
        st.markdown("**2. Area Sehat (Non-Distress)**")
        bg_mode = st.radio("Metode Input Area Sehat:", ["Gunakan Poligon AOI (.zip)", "Gunakan Orthomosaic (.tif)"])
        aoi_bg_file = None; ortho_file = None; ortho_link = ""
        
        if bg_mode == "Gunakan Poligon AOI (.zip)":
            aoi_bg_file = st.file_uploader("Upload SHP Area Sehat (.zip)", type="zip", key="aoi_bg")
        else:
            ortho_mode = st.radio("Sumber Ortho:", ["Paste Link Google Drive", "Upload File .tif"], key="ortho_m")
            if ortho_mode == "Upload File .tif": ortho_file = st.file_uploader("Upload Ortho (.tif)", type="tif", key="ortho_f")
            else: ortho_link = st.text_input("Link Drive Ortho (.tif)", key="ortho_l")

    if st.button("🎯 Generate Random Sample", type="primary", use_container_width=True):
        konfigurasi_sampling = {
            "Alligator Crack": file_ac_s, "Edge Crack": file_ec_s, "Longitudinal Crack": file_lc_s,
            "Potholes": file_pt_s, "Patching": file_pa_s, "Rutting": file_rt_s
        }
        
        with st.spinner("Mengeksekusi algoritma sampling terarah..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    kumpulan_titik = []
                    max_distress_samples = 0
                    
                    for jenis, file in konfigurasi_sampling.items():
                        if file is not None:
                            gdf = read_zip_shapefile(file, tmpdir)
                            if gdf is not None and not gdf.empty:
                                if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
                                gdf = gdf.to_crs(epsg=epsg_code)
                                gdf_valid = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]
                                N_pop = len(gdf_valid)
                                
                                if metode_sampling in ["Input Manual", "Aturan Roscoe"]: target_n = n_manual
                                elif metode_sampling == "Rumus Slovin": target_n = math.ceil(N_pop / (1 + (N_pop * (e_slovin**2)))) if N_pop > 0 else 0
                                elif metode_sampling == "Persentase Populasi": target_n = math.ceil(N_pop * p_persen)
                                
                                actual_samples_taken = min(target_n, N_pop)
                                if actual_samples_taken > max_distress_samples: max_distress_samples = actual_samples_taken
                                
                                points = get_random_points_gdf(gdf_valid, target_n, is_background=False)
                                for p in points: kumpulan_titik.append({"Class": jenis, "geometry": p})

                    points_bg = []
                    target_bg = max_distress_samples if max_distress_samples > 0 else (n_manual if metode_sampling in ["Input Manual", "Aturan Roscoe"] else 50)
                    
                    if bg_mode == "Gunakan Poligon AOI (.zip)":
                        if aoi_bg_file is not None:
                            aoi_gdf = read_zip_shapefile(aoi_bg_file, tmpdir)
                            if aoi_gdf is not None and not aoi_gdf.empty:
                                aoi_gdf = aoi_gdf.to_crs(epsg=epsg_code)
                                points_bg = get_random_points_gdf(aoi_gdf, target_bg, is_background=True)
                    else:
                        is_ortho_valid = (ortho_mode == "Upload File .tif" and ortho_file is not None) or (ortho_mode == "Paste Link Google Drive" and ortho_link != "")
                        if is_ortho_valid:
                            ortho_path = os.path.join(tmpdir, "ortho.tif")
                            if ortho_mode == "Upload File .tif":
                                with open(ortho_path, "wb") as f: f.write(ortho_file.getbuffer())
                            else:
                                import gdown, re
                                match = re.search(r"/d/([a-zA-Z0-9_-]+)", ortho_link)
                                if match: gdown.download(id=match.group(1), output=ortho_path, quiet=False)
                            points_bg = get_random_points_raster(ortho_path, target_bg, epsg_code)
                            
                    for p in points_bg: kumpulan_titik.append({"Class": "Non-Distress", "geometry": p})

                    if not kumpulan_titik:
                        st.error("❌ Tidak ada input yang dimasukkan atau diproses.")
                        st.stop()

                    gdf_sample = gpd.GeoDataFrame(kumpulan_titik, crs=f"EPSG:{epsg_code}")
                    gdf_sample["deteksi"] = gdf_sample["Class"].apply(assign_deteksi_class)
                    gdf_sample["aktual"] = ""
                    
                    stat_sample = gdf_sample["Class"].value_counts().reset_index()
                    stat_sample.columns = ["Kelas / Kategori", "Jumlah Sampel Diekstrak"]
                    
                    gpkg_samp_path = os.path.join(tmpdir, "GroundTruth_Samples.gpkg")
                    gdf_sample.to_file(gpkg_samp_path, driver="GPKG")
                    
                    st.session_state.df_stat_samp = stat_sample
                    with open(gpkg_samp_path, "rb") as f: st.session_state.gpkg_bytes_samp = f.read()
                    st.session_state.sampling_selesai = True
                    
                except Exception as e:
                    st.error(f"❌ Kesalahan: {e}"); st.session_state.sampling_selesai = False

    if st.session_state.get('sampling_selesai', False):
        st.success("✅ Generasi Random Samples berhasil!")
        st.dataframe(st.session_state.df_stat_samp, use_container_width=True, hide_index=True)
        st.download_button(
            "🗺️ Download GPKG Titik Sampel", data=st.session_state.gpkg_bytes_samp, 
            file_name="GroundTruth_Samples.gpkg", mime="application/geopackage+sqlite3", 
            type="primary", use_container_width=True
        )

# ==========================================
# MODUL 3: EVALUASI AKURASI (CONFUSION MATRIX)
# ==========================================
elif menu == "📊 Evaluasi Akurasi (Confusion Matrix)":
    st.subheader("Modul Evaluasi Akurasi (Confusion Matrix)")
    st.markdown("Unggah file Excel hasil survei lapangan yang telah diisi nilainya pada kolom **'aktual'**.")
    
    # --- TAMBAHAN PANDUAN & TEMPLATE UNTUK PEMULA ---
    with st.expander("💡 Klik untuk melihat Contoh Format Excel & Download Template"):
        st.markdown(
            "Sistem mewajibkan adanya kolom **`deteksi`** (hasil prediksi model) dan **`aktual`** (kebenaran dari lapangan). "
            "Isilah kolom **`aktual`** dengan angka **1 hingga 7** sesuai keterangan kelas kerusakan."
        )
        
        # Bikin DataFrame Contoh
        contoh_df = pd.DataFrame({
            "FID": [0, 1, 2, 3, 4],
            "Class": ["Alligator Crack", "Edge Crack", "Potholes", "Rutting", "Non-Distress"],
            "deteksi": [1, 2, 5, 6, 7],
            "aktual": [1, 2, 7, 6, 7] 
        })
        
        # Tampilkan tabel contoh
        st.dataframe(contoh_df, use_container_width=True, hide_index=True)
        st.caption("*Catatan: Pada baris ke-3 (Potholes), nilai aktual diisi 7 (Non-Distress) yang berarti model salah mendeteksi di titik tersebut.*")
        
        # Buat file template Excel di memori untuk diunduh
        template_buffer = io.BytesIO()
        with pd.ExcelWriter(template_buffer, engine='xlsxwriter') as writer:
            contoh_df.to_excel(writer, index=False)
            
        st.download_button(
            label="📥 Download Template Excel",
            data=template_buffer.getvalue(),
            file_name="Template_Evaluasi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    # ------------------------------------------------

    if 'eval_selesai' not in st.session_state: st.session_state.eval_selesai = False

    # Izinkan xls dan xlsx
    file_eval = st.file_uploader("Upload File Excel (.xls, .xlsx)", type=["xls", "xlsx"])
    
    if st.button("🎯 Proses Evaluasi Data", type="primary", use_container_width=True):
        if file_eval is not None:
            with st.spinner("Menghitung metrik geospasial dan membuat visualisasi..."):
                try:
                    df = pd.read_excel(file_eval)
                    
                    if 'deteksi' not in df.columns or 'aktual' not in df.columns:
                        st.error("❌ File Excel harus memiliki setidaknya kolom bernama 'deteksi' dan 'aktual'. Silakan download template sebagai panduan.")
                        st.stop()
                    
                    df_clean = df.dropna(subset=['aktual', 'deteksi']).copy()
                    df_clean['deteksi'] = pd.to_numeric(df_clean['deteksi'], errors='coerce').astype('Int64')
                    df_clean['aktual'] = pd.to_numeric(df_clean['aktual'], errors='coerce').astype('Int64')
                    df_clean = df_clean.dropna(subset=['aktual', 'deteksi']) 
                    
                    if df_clean.empty:
                        st.warning("⚠️ Tidak ada data valid untuk dievaluasi. Pastikan kolom 'aktual' di Excel sudah Anda ketik dengan angka (1-7).")
                        st.stop()

                    label_mapping = {
                        1: "Alligator Crack", 2: "Edge Crack", 3: "Longitudinal Crack",
                        4: "Patching", 5: "Potholes", 6: "Rutting", 7: "Non-Distress"
                    }
                    
                    unique_classes = sorted(list(set(df_clean['aktual'].dropna()) | set(df_clean['deteksi'].dropna())))
                    target_names = [label_mapping.get(int(i), f"Kelas {i}") for i in unique_classes]
                    
                    # 1. Hitung Confusion Matrix Dasar
                    cm_data = confusion_matrix(df_clean['aktual'], df_clean['deteksi'], labels=unique_classes)
                    
                    # 2. Kalkulasi Metrik Global (Overall, Expected Agreement, Kappa)
                    po = accuracy_score(df_clean['aktual'], df_clean['deteksi']) 
                    total_samples = np.sum(cm_data)
                    sum_rows = np.sum(cm_data, axis=1) 
                    sum_cols = np.sum(cm_data, axis=0) 
                    
                    pe = np.sum((sum_rows * sum_cols) / (total_samples ** 2))
                    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
                    
                    # 3. Generate Matriks Image (PNG)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=target_names, yticklabels=target_names, ax=ax)
                    ax.set_xlabel("Prediksi Model (Deteksi)")
                    ax.set_ylabel("Kebenaran Lapangan (Aktual)")
                    plt.title("Confusion Matrix", pad=20)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    png_buffer = io.BytesIO()
                    fig.savefig(png_buffer, format="png", dpi=300)
                    png_buffer.seek(0)
                    
                    # 4. Classification Report (Translasi ke Istilah Geospasial)
                    report_dict = classification_report(df_clean['aktual'], df_clean['deteksi'], 
                                                        labels=unique_classes, target_names=target_names, 
                                                        output_dict=True, zero_division=0)
                    df_report = pd.DataFrame(report_dict).transpose()
                    
                    df_report.rename(columns={
                        'precision': "User's Acc (Precision)",
                        'recall': "Producer's Acc (Recall)",
                        'f1-score': "F1-Score",
                        'support': "Support (Jumlah)"
                    }, inplace=True)
                    
                    # Generate Excel Bawaan Laporan
                    excel_eval_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_eval_buffer, engine='xlsxwriter') as writer:
                        df_report.to_excel(writer, sheet_name='Classification_Report')
                        df_clean.to_excel(writer, sheet_name='Data_Clean', index=False)
                    excel_eval_buffer.seek(0)
                    
                    # Generate PDF Report
                    with tempfile.TemporaryDirectory() as tmpdir:
                        cm_path = os.path.join(tmpdir, "cm.png")
                        fig.savefig(cm_path, format="png", dpi=300)
                        
                        pdf_path = os.path.join(tmpdir, "Evaluasi_Laporan.pdf")
                        generate_pdf_eval_report(df_report.copy(), po, cm_path, pdf_path)
                        
                        with open(pdf_path, "rb") as f:
                            pdf_eval_bytes = f.read()

                    # Simpan State
                    st.session_state.eval_po = po
                    st.session_state.eval_pe = pe
                    st.session_state.eval_kappa = kappa
                    st.session_state.eval_df_report = df_report
                    st.session_state.eval_png = png_buffer.getvalue()
                    st.session_state.eval_excel = excel_eval_buffer.getvalue()
                    st.session_state.eval_pdf = pdf_eval_bytes
                    st.session_state.eval_selesai = True
                    st.session_state.eval_jumlah_data = len(df_clean)

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
                    st.session_state.eval_selesai = False
        else:
            st.warning("⚠️ Harap unggah file Excel terlebih dahulu.")

    # Tampilkan Hasil Evaluasi jika sukses
    if st.session_state.get('eval_selesai', False):
        st.success(f"✅ Berhasil memuat {st.session_state.eval_jumlah_data} baris data valid untuk dievaluasi.")
        st.divider()
        
        # --- METRIK GLOBAL ---
        st.markdown("### 📈 Ringkasan Metrik Akurasi Global")
        metrik_col1, metrik_col2, metrik_col3 = st.columns(3)
        metrik_col1.metric("Overall Accuracy (Po)", f"{st.session_state.eval_po:.2%}")
        metrik_col2.metric("Expected Agreement (Pe)", f"{st.session_state.eval_pe:.4f}")
        
        # Menentukan status reliabilitas Kappa
        kappa_val = st.session_state.eval_kappa
        if kappa_val > 0.80: kappa_status = "Sangat Kuat"
        elif kappa_val > 0.60: kappa_status = "Kuat"
        elif kappa_val > 0.40: kappa_status = "Moderat"
        elif kappa_val > 0.20: kappa_status = "Cukup"
        else: kappa_status = "Lemah"
        
        metrik_col3.metric("Kappa Coefficient", f"{kappa_val:.4f}", f"Reliabilitas: {kappa_status}")
        
        st.divider()
        
        # --- MATRIKS CONFUSION ---
        st.markdown("### 📊 Heatmap Confusion Matrix")
        st.image(st.session_state.eval_png, use_container_width=False)
        
        st.divider()
        
        # --- LAPORAN PER KELAS ---
        st.markdown("### 📋 Laporan Klasifikasi Detail (Per Kelas)")
        st.caption("Catatan: User's Accuracy identik dengan Precision. Producer's Accuracy identik dengan Recall.")
        
        df_display = st.session_state.eval_df_report.style.format({
            "User's Acc (Precision)": "{:.2f}", 
            "Producer's Acc (Recall)": "{:.2f}", 
            "F1-Score": "{:.2f}", 
            "Support (Jumlah)": "{:.0f}"
        })
        st.dataframe(df_display, use_container_width=True)
        
        st.divider()
        st.markdown("### 📥 Unduh Hasil Evaluasi")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1: 
            st.download_button("🖼️ PNG Matriks", data=st.session_state.eval_png, file_name="Confusion_Matrix.png", mime="image/png", use_container_width=True)
        with col_dl2: 
            st.download_button("📊 Laporan Excel", data=st.session_state.eval_excel, file_name="Evaluasi_Metrik.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col_dl3: 
            st.download_button("📄 Laporan PDF", data=st.session_state.eval_pdf, file_name="Evaluasi_Laporan.pdf", mime="application/pdf", use_container_width=True)
