import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
from shapely.geometry import LineString
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
st.set_page_config(page_title="ASPAL GeoExt", page_icon="📏", layout="wide")

# ==========================================
# 2. FUNGSI SPASIAL & MATEMATIKA
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

def hitung_geometri_dasar(gdf):
    """Menghitung Panjang, Lebar, Diameter, dan Luas (Metrik 1, 2, 3, 4)"""
    gdf = gdf.copy()
    panjang, lebar, diameter, luas = [], [], [], []
    
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            panjang.append(0); lebar.append(0); diameter.append(0); luas.append(0)
            continue
            
        # Menggunakan Minimum Rotated Rectangle untuk estimasi panjang & lebar
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [LineString([coords[i], coords[i+1]]).length for i in range(4)]
        
        # Ambil sisi unik untuk menentukan panjang dan lebar
        unique_edges = sorted(list(set([round(e, 5) for e in edges])))
        p = max(unique_edges) if unique_edges else 0
        l = min(unique_edges) if unique_edges else 0
        d = (p + l) / 2 if len(unique_edges) >= 2 else p # Pendekatan diameter
        
        panjang.append(p)
        lebar.append(l)
        diameter.append(d)
        luas.append(geom.area)
        
    gdf["Panjang_m"] = np.round(panjang, 3)
    gdf["Lebar_m"] = np.round(lebar, 3)
    gdf["Diameter_m"] = np.round(diameter, 3)
    gdf["Luas_m2"] = np.round(luas, 3)
    return gdf

def hitung_kedalaman(gdf, dsm_path, buffer_distance=0.3):
    """Menghitung Kedalaman (Metrik 5) menggunakan DSM"""
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
        
        if z_hole is not None and z_ref is not None:
            depth = (z_ref - z_hole) # dalam satuan meter (mengikuti resolusi DSM)
            depth = max(0, depth)
        else:
            depth = 0
        kedalaman.append(depth)
        
    gdf["Kedalaman_m"] = np.round(kedalaman, 4)
    return gdf

def generate_pdf_report(df_rekap, pdf_path):
    """Membuat laporan PDF standar ASPAL GeoExt"""
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<b>LAPORAN EKSTRAKSI GEOMETRI</b>", styles['Title']))
    elements.append(Paragraph("<b>ASPAL GeoExt (Geometry Extractor)</b>", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))
    
    # Konversi DataFrame ke format list untuk ReportLab Table
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
    
    elements.append(t)
    doc.build(elements)

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
# Hero Section
st.markdown("<h1 style='text-align: center; color: #4da6ff;'>📐 ASPAL GeoExt</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #cbd5e1;'>Sub-sistem Ekstraksi Geometri Kerusakan Jalan Berbasis GIS & UAV</h4>", unsafe_allow_html=True)
st.divider()

# Inisialisasi Session State
if 'geoext_selesai' not in st.session_state: st.session_state.geoext_selesai = False

with st.sidebar:
    st.header("⚙️ Pengaturan ASPAL GeoExt")
    epsg_code = st.number_input("Kode EPSG Proyeksi (UTM Lokal)", value=32749, step=1, help="Penting agar hasil luasan/panjang memiliki akurasi dalam satuan meter.")
    st.caption("Contoh: 32749 untuk UTM Zone 49S")
    
    st.divider()
    st.markdown("### 🛠️ Parameter Ekstraksi")
    st.markdown("""
    Sistem akan mengekstrak metrik berikut:
    1. **Panjang (m)**
    2. **Lebar (m)**
    3. **Diameter (m)**
    4. **Luas (m²)**
    5. **Kedalaman (m)** *(Membutuhkan DSM)*
    """)
    
    if st.button("🔄 Reset Modul", use_container_width=True):
        st.session_state.clear()
        st.rerun()

col_input1, col_input2 = st.columns([1.5, 1])

with col_input1:
    st.subheader("📁 1. Input Shapefile Kerusakan")
    st.info("Unggah shapefile poligon (.zip) untuk masing-masing jenis kerusakan yang ditemukan di lapangan.")
    
    c1, c2 = st.columns(2)
    with c1:
        file_ac = st.file_uploader("1. Alligator Crack", type="zip", help="Metrik: P, L, D, Luas")
        file_ec = st.file_uploader("2. Edge Crack", type="zip", help="Metrik: P, L, D, Luas")
        file_lc = st.file_uploader("3. Longitudinal Crack", type="zip", help="Metrik: P, L, D, Luas")
    with c2:
        file_pt = st.file_uploader("4. Potholes", type="zip", help="Metrik: P, L, D, Luas, Kedalaman")
        file_pa = st.file_uploader("5. Patching", type="zip", help="Metrik: P, L, D, Luas")
        file_rt = st.file_uploader("6. Rutting", type="zip", help="Metrik: P, L, D, Luas, Kedalaman")

with col_input2:
    st.subheader("🗺️ 2. Input Data Kedalaman")
    st.warning("Data **Digital Surface Model (DSM)** diperlukan untuk mengekstrak dimensi kedalaman pada kerusakan tipe *Potholes* dan *Rutting*.")
    dsm_file = st.file_uploader("Upload DSM (.tif)", type="tif")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. PROSES EKSTRAKSI
# ==========================================
if st.button("🚀 Ekstrak Geometri Kerusakan", type="primary", use_container_width=True):
    konfigurasi_input = {
        "Alligator_Crack": {"file": file_ac, "kebutuhan_dsm": False},
        "Edge_Crack": {"file": file_ec, "kebutuhan_dsm": False},
        "Longitudinal_Crack": {"file": file_lc, "kebutuhan_dsm": False},
        "Potholes": {"file": file_pt, "kebutuhan_dsm": True},
        "Patching": {"file": file_pa, "kebutuhan_dsm": False},
        "Rutting": {"file": file_rt, "kebutuhan_dsm": True}
    }
    
    with st.spinner("Mengaktifkan modul spasial ASPAL GeoExt... Memproses perhitungan geometri..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Simpan DSM jika ada
                dsm_path = None
                if dsm_file:
                    dsm_path = os.path.join(tmpdir, "dsm.tif")
                    with open(dsm_path, "wb") as f:
                        f.write(dsm_file.getbuffer())

                hasil_gdf_list = []
                rekap_data = []

                for jenis_kerusakan, config in konfigurasi_input.items():
                    file = config["file"]
                    butuh_dsm = config["kebutuhan_dsm"]
                    
                    if file is not None:
                        gdf = read_zip_shapefile(file, tmpdir)
                        if gdf is not None and not gdf.empty:
                            # Set/Transform CRS ke satuan meter
                            if gdf.crs is None:
                                gdf.set_crs(epsg=4326, inplace=True)
                            gdf = gdf.to_crs(epsg=epsg_code)
                            
                            # Hitung Metrik 1, 2, 3, 4 (Semua kerusakan)
                            gdf = hitung_geometri_dasar(gdf)
                            
                            # Kolom default kedalaman
                            gdf["Kedalaman_m"] = 0.0
                            
                            # Hitung Metrik 5 (Khusus Potholes & Rutting jika ada DSM)
                            if butuh_dsm:
                                if dsm_path:
                                    gdf = hitung_kedalaman(gdf, dsm_path)
                                else:
                                    st.warning(f"⚠️ Data DSM tidak diunggah. Nilai Kedalaman untuk {jenis_kerusakan.replace('_', ' ')} otomatis di-set ke 0.")
                            
                            gdf["Jenis_Kerusakan"] = jenis_kerusakan.replace("_", " ")
                            hasil_gdf_list.append(gdf)
                            
                            # Masukkan ke tabel rekap (Ambil rata-rata)
                            rekap_data.append({
                                "Jenis Kerusakan": jenis_kerusakan.replace("_", " "),
                                "Total Objek": len(gdf),
                                "Rata Panjang (m)": round(gdf["Panjang_m"].mean(), 3),
                                "Rata Lebar (m)": round(gdf["Lebar_m"].mean(), 3),
                                "Rata Diameter (m)": round(gdf["Diameter_m"].mean(), 3),
                                "Total Luas (m²)": round(gdf["Luas_m2"].sum(), 3),
                                "Rata Kedalaman (m)": round(gdf["Kedalaman_m"].mean(), 3) if butuh_dsm else "-"
                            })

                if not hasil_gdf_list:
                    st.error("❌ Analisis dihentikan: Tidak ada file Shapefile valid yang diunggah. Mohon unggah minimal 1 jenis kerusakan.")
                    st.stop()

                # Gabungkan semua data spasial
                master_gdf = pd.concat(hasil_gdf_list, ignore_index=True)
                df_rekap = pd.DataFrame(rekap_data)

                # Export Output
                gpkg_path = os.path.join(tmpdir, "ASPAL_GeoExt_Hasil.gpkg")
                master_gdf.to_file(gpkg_path, driver="GPKG")
                
                excel_buffer = io.BytesIO()
                # Drop kolom geometry agar bisa disave ke Excel standar
                df_detail = master_gdf.drop(columns=['geometry'])
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df_rekap.to_excel(writer, sheet_name='Rekapitulasi', index=False)
                    df_detail.to_excel(writer, sheet_name='Detail Ekstraksi', index=False)
                
                pdf_path = os.path.join(tmpdir, "ASPAL_GeoExt_Laporan.pdf")
                generate_pdf_report(df_rekap, pdf_path)

                # Simpan di Session State
                st.session_state.df_rekap_geo = df_rekap
                st.session_state.df_detail_geo = df_detail
                with open(gpkg_path, "rb") as f: st.session_state.gpkg_bytes_geo = f.read()
                st.session_state.excel_bytes_geo = excel_buffer.getvalue()
                with open(pdf_path, "rb") as f: st.session_state.pdf_bytes_geo = f.read()
                st.session_state.geoext_selesai = True

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat ekstraksi spasial: {e}")
                st.session_state.geoext_selesai = False

# ==========================================
# 5. HASIL & DOWNLOAD
# ==========================================
if st.session_state.geoext_selesai:
    st.success("✅ Ekstraksi geometri berhasil diselesaikan oleh ASPAL GeoExt!")
    
    st.markdown("---")
    st.subheader("📋 Tabel Rekapitulasi Ekstraksi")
    st.dataframe(st.session_state.df_rekap_geo, use_container_width=True, hide_index=True)
    
    with st.expander("Lihat Detail Ekstraksi Per Objek Poligon"):
        st.dataframe(st.session_state.df_detail_geo, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💾 Unduh Hasil Analisis ASPAL GeoExt")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button("📄 Laporan Rekapitulasi (.pdf)", data=st.session_state.pdf_bytes_geo, file_name="ASPAL_GeoExt_Laporan.pdf", mime="application/pdf", type="primary", use_container_width=True)
    with col_dl2:
        st.download_button("📊 Data Ekstraksi Lengkap (.xlsx)", data=st.session_state.excel_bytes_geo, file_name="ASPAL_GeoExt_Data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with col_dl3:
        st.download_button("🗺️ Peta Atribut Spasial (.gpkg)", data=st.session_state.gpkg_bytes_geo, file_name="ASPAL_GeoExt_Peta.gpkg", mime="application/geopackage+sqlite3", use_container_width=True)