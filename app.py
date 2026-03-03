import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import random
from shapely.geometry import Point, LineString
import rasterio
from rasterstats import zonal_stats
import io

# Import untuk ReportLab (PDF)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =========================================
# 1. KONFIGURASI HALAMAN UTAMA
# =========================================
st.set_page_config(page_title="ASPAL GeoExt", page_icon="📐", layout="wide")

# ==========================================
# 2. FUNGSI SPASIAL (SHARED)
# ==========================================
def read_zip_shapefile(uploaded_file, tmpdir):
    """Membaca file shapefile dari dalam format .zip"""
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

# ==========================================
# FUNGSI MODUL EKSTRAKSI GEOMETRI
# ==========================================
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

# ==========================================
# FUNGSI MODUL RANDOM SAMPLING
# ==========================================
def get_random_points_gdf(gdf, n_points, epsg_code):
    """Mencari titik acak yang jatuh TEPAT DI DALAM poligon shapefile"""
    gdf = gdf.to_crs(epsg=epsg_code)
    union_geom = gdf.geometry.union_all()
    points = []
    
    # Jika geometri kosong
    if union_geom.is_empty: return []
    
    minx, miny, maxx, maxy = union_geom.bounds
    attempts = 0
    max_attempts = n_points * 100 # Batas aman agar tidak infinite loop
    
    while len(points) < n_points and attempts < max_attempts:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if union_geom.contains(p):
            points.append(p)
        attempts += 1
        
    return points

def get_random_points_raster(raster_path, n_points, epsg_code):
    """Mencari titik acak dari area valid Orthomosaic (bukan NoData/Background hitam)"""
    points = []
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_crs = src.crs
        nodata = src.nodata
        
        attempts = 0
        max_attempts = n_points * 100
        
        while len(points) < n_points and attempts < max_attempts:
            x = random.uniform(bounds.left, bounds.right)
            y = random.uniform(bounds.bottom, bounds.top)
            
            try:
                # Cek piksel di koordinat tersebut
                val = next(src.sample([(x, y)]))
                # Pastikan bukan Nodata dan bukan full 0 (hitam)
                if (nodata is None or val[0] != nodata) and np.any(val > 0):
                    points.append(Point(x, y))
            except:
                pass
            attempts += 1
            
    # Konversi ke CRS target jika berbeda
    if points:
        gdf_temp = gpd.GeoDataFrame(geometry=points, crs=raster_crs)
        return gdf_temp.to_crs(epsg=epsg_code).geometry.tolist()
    return []

# ==========================================
# 3. SIDEBAR & NAVIGASI
# ==========================================
st.markdown("<h1 style='text-align: center; color: #4da6ff;'>📐 ASPAL GeoExt</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #cbd5e1;'>Sistem Ekstraksi Geometri & Sampling Acak Berbasis GIS</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("🧭 Navigasi Modul")
    menu = st.radio("Pilih Modul:", [
        "📏 Ekstraksi Geometri Kerusakan",
        "🎯 Random Sampling (Ground Truth)"
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
            "Alligator_Crack": {"file": file_ac, "kebutuhan_dsm": False},
            "Edge_Crack": {"file": file_ec, "kebutuhan_dsm": False},
            "Longitudinal_Crack": {"file": file_lc, "kebutuhan_dsm": False},
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
                                gdf["Jenis_Kerusakan"] = jk.replace("_", " ")
                                hasil_gdf_list.append(gdf)
                                rekap_data.append({
                                    "Jenis Kerusakan": jk.replace("_", " "), "Total Objek": len(gdf),
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
    st.info("Pilih jumlah sampel (titik) yang ingin Anda hasilkan secara acak dari masing-masing klasifikasi kerusakan serta area sehat (Orthomosaic).")
    
    if 'sampling_selesai' not in st.session_state: st.session_state.sampling_selesai = False

    n_samples = st.number_input("Tentukan Jumlah Sampel Titik per Kelas", min_value=1, max_value=1000, value=50, step=10)
    
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
        st.markdown("**2. Orthomosaic (Area Sehat/Background)**")
        st.caption("Digunakan untuk mengekstrak titik acak sebagai kelas negatif (jalan sehat).")
        ortho_mode = st.radio("Input Orthomosaic:", ["Paste Link Google Drive", "Upload File .tif"], key="ortho_m")
        ortho_file = None; ortho_link = ""
        if ortho_mode == "Upload File .tif": ortho_file = st.file_uploader("Upload Ortho (.tif)", type="tif", key="ortho_f")
        else: ortho_link = st.text_input("Link Drive Ortho (.tif)", key="ortho_l")

    if st.button("🎯 Buat Random Sample", type="primary", use_container_width=True):
        konfigurasi_sampling = {
            "Alligator Crack": file_ac_s,
            "Edge Crack": file_ec_s,
            "Longitudinal Crack": file_lc_s,
            "Potholes": file_pt_s,
            "Patching": file_pa_s,
            "Rutting": file_rt_s
        }
        
        with st.spinner(f"Mencari {n_samples} titik acak dari masing-masing input..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    kumpulan_titik = []
                    
                    # 1. Proses Vector / Kerusakan
                    for jenis, file in konfigurasi_sampling.items():
                        if file is not None:
                            gdf = read_zip_shapefile(file, tmpdir)
                            if gdf is not None and not gdf.empty:
                                if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
                                points = get_random_points_gdf(gdf, n_samples, epsg_code)
                                for p in points:
                                    kumpulan_titik.append({"Class": jenis, "geometry": p})

                    # 2. Proses Raster / Orthomosaic (Background)
                    is_ortho_valid = (ortho_mode == "Upload File .tif" and ortho_file is not None) or (ortho_mode == "Paste Link Google Drive" and ortho_link != "")
                    if is_ortho_valid:
                        ortho_path = os.path.join(tmpdir, "ortho.tif")
                        if ortho_mode == "Upload File .tif":
                            with open(ortho_path, "wb") as f: f.write(ortho_file.getbuffer())
                        else:
                            import gdown, re
                            match = re.search(r"/d/([a-zA-Z0-9_-]+)", ortho_link)
                            if match: gdown.download(id=match.group(1), output=ortho_path, quiet=False)
                        
                        points_bg = get_random_points_raster(ortho_path, n_samples, epsg_code)
                        for p in points_bg:
                            kumpulan_titik.append({"Class": "Background_Sehat", "geometry": p})

                    if not kumpulan_titik:
                        st.error("❌ Tidak ada input yang dimasukkan.")
                        st.stop()

                    # Jadikan GeoDataFrame Gabungan
                    gdf_sample = gpd.GeoDataFrame(kumpulan_titik, crs=f"EPSG:{epsg_code}")
                    
                    # Generate Statistik 
                    stat_sample = gdf_sample["Class"].value_counts().reset_index()
                    stat_sample.columns = ["Kelas / Kategori", "Jumlah Sampel Didapat"]
                    
                    # Export ke GPKG
                    gpkg_samp_path = os.path.join(tmpdir, "Random_Sample_GroundTruth.gpkg")
                    gdf_sample.to_file(gpkg_samp_path, driver="GPKG")
                    
                    st.session_state.df_stat_samp = stat_sample
                    with open(gpkg_samp_path, "rb") as f: st.session_state.gpkg_bytes_samp = f.read()
                    st.session_state.sampling_selesai = True
                    
                except Exception as e:
                    st.error(f"❌ Kesalahan saat sampling: {e}")
                    st.session_state.sampling_selesai = False

    if st.session_state.get('sampling_selesai', False):
        st.success("✅ Titik acak (Random Samples) berhasil digenerate!")
        st.markdown("Berikut adalah ringkasan sampel yang berhasil diekstrak (Mungkin ada yang kurang dari target jika area poligonnya sangat kecil):")
        
        st.dataframe(st.session_state.df_stat_samp, use_container_width=True, hide_index=True)
        
        st.download_button(
            "🗺️ Download Shapefile GPKG Titik Sampel", 
            data=st.session_state.gpkg_bytes_samp, 
            file_name="Random_Samples.gpkg", 
            mime="application/geopackage+sqlite3", 
            type="primary", 
            use_container_width=True
        )
