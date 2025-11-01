import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta

# Mengatur konfigurasi halaman Streamlit agar lebih lebar
st.set_page_config(layout="wide")

# =============================================================================
# FUNGSI DATA LOADING DAN CLEANING (Berdasarkan Notebook)
# =============================================================================
@st.cache_data
def load_and_prep_data():
    """
    Memuat, membersihkan, dan menggabungkan semua data dari notebook.
    Ini adalah versi gabungan dari sel Data Wrangling, Cleaning, dan EDA.
    """
    # 1. Load semua dataset (sesuai sel Gathering Data)
    try:
        customers_df = pd.read_csv('./data/customers_dataset.csv')
        orders_df = pd.read_csv('./data/orders_dataset.csv')
        order_items_df = pd.read_csv('./data/order_items_dataset.csv')
        order_payments_df = pd.read_csv('./data/order_payments_dataset.csv')
        products_df = pd.read_csv('./data/products_dataset.csv')
        translation_df = pd.read_csv('./data/product_category_name_translation.csv')
    except FileNotFoundError:
        st.error("Pastikan semua file .csv berada di dalam folder 'data/'.")
        return None, None, None

    # 2. Cleaning Data (sesuai sel Cleaning Data)
    
    # Membersihkan orders_df (sesuai sel 43-45)
    # Menghapus baris dengan data pengiriman/approval yang hilang
    orders_clean_df = orders_df.dropna(subset=[
        'order_approved_at', 
        'order_delivered_carrier_date', 
        'order_delivered_customer_date'
    ])
    
    # Konversi kolom timestamp (sesuai sel 44)
    timestamp_cols = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in timestamp_cols:
        orders_clean_df[col] = pd.to_datetime(orders_clean_df[col])

    # 3. Persiapan Data untuk Dashboard
    
    # --- DataFrame 1: orders_main_df (1 baris per PESANAN) ---
    # Digunakan untuk analisis Pendapatan dan Pesanan
    
    # Agregasi pembayaran per order_id
    payments_agg_df = order_payments_df.groupby('order_id')['payment_value'].sum().reset_index()

    # Gabungkan Orders + Customers + Payments
    orders_main_df = orders_clean_df.merge(customers_df, on='customer_id', how='left')
    orders_main_df = orders_main_df.merge(payments_agg_df, on='order_id', how='left')
    
    # Hapus pesanan tanpa info pembayaran setelah merge
    orders_main_df.dropna(subset=['payment_value'], inplace=True)

    # Feature Engineering (sesuai sel 64-67)
    orders_main_df['order_purchase_year'] = orders_main_df['order_purchase_timestamp'].dt.year
    orders_main_df['order_purchase_month'] = orders_main_df['order_purchase_timestamp'].dt.month
    orders_main_df['order_purchase_month_name'] = orders_main_df['order_purchase_timestamp'].dt.month_name()
    orders_main_df['order_purchase_quarter'] = orders_main_df['order_purchase_timestamp'].dt.quarter
    orders_main_df['order_purchase_month_year'] = orders_main_df['order_purchase_timestamp'].dt.to_period('M')

    
    # --- DataFrame 2: full_items_df (1 baris per ITEM) ---
    # Digunakan untuk analisis Produk/Kategori dan RFM (karena butuh 'price')

    # Gabungkan produk dengan nama terjemahan (Improvement dari notebook)
    products_en_df = products_df.merge(translation_df, on='product_category_name', how='left')
    
    # Gabungkan items dengan info produk
    items_products_df = order_items_df.merge(products_en_df, on='product_id', how='left')
    
    # Gabungkan dengan data order utama
    full_items_df = orders_main_df.merge(items_products_df, on='order_id', how='left')
    
    # Hapus item tanpa info produk/harga setelah merge
    full_items_df.dropna(subset=['product_category_name_english', 'price'], inplace=True)

    return orders_main_df, full_items_df, customers_df

# =============================================================================
# Load Data
# =============================================================================

orders_main_df, full_items_df, customers_df = load_and_prep_data()

if orders_main_df is not None:

    # =============================================================================
    # SIDEBAR UNTUK FILTER
    # =============================================================================
    
    st.sidebar.title("Filter Dashboard")

    # Filter Tahun
    available_years = sorted(orders_main_df['order_purchase_year'].unique())
    selected_year = st.sidebar.multiselect(
        "Pilih Tahun",
        options=available_years,
        default=available_years
    )

    # Filter Kuartal
    available_quarters = sorted(orders_main_df['order_purchase_quarter'].unique())
    selected_quarter = st.sidebar.multiselect(
        "Pilih Kuartal",
        options=available_quarters,
        default=available_quarters
    )

    # Terapkan Filter
    filtered_orders_df = orders_main_df[
        (orders_main_df['order_purchase_year'].isin(selected_year)) &
        (orders_main_df['order_purchase_quarter'].isin(selected_quarter))
    ]

    filtered_items_df = full_items_df[
        (full_items_df['order_purchase_year'].isin(selected_year)) &
        (full_items_df['order_purchase_quarter'].isin(selected_quarter))
    ]

    # =============================================================================
    # HALAMAN UTAMA (MAIN PAGE)
    # =============================================================================
    
    st.title("Dashboard Analisis E-Commerce")
    st.markdown("Dashboard ini menampilkan analisis dari E-Commerce Public Dataset berdasarkan notebook Proyek Analisis Data.")

    # Definisikan Tab
    tab_ringkasan, tab_pendapatan, tab_produk, tab_pelanggan = st.tabs(
        ["Ringkasan", "Analisis Pendapatan & Pesanan", "Analisis Produk", "Analisis Pelanggan"]
    )

    # --- TAB 1: Ringkasan ---
    with tab_ringkasan:
        st.header("Ringkasan Eksekutif (Berdasarkan Filter)")
        
        # Metrik Utama
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pendapatan (Payment)", f"R$ {filtered_orders_df['payment_value'].sum():,.2f}")
        col2.metric("Total Pesanan", f"{filtered_orders_df['order_id'].nunique():,}")
        col3.metric("Pelanggan Unik", f"{filtered_orders_df['customer_unique_id'].nunique():,}")
        
        st.markdown("---")
        
        st.header("Tren Penjualan Bulanan (All Time)")
        st.markdown("Visualisasi ini menunjukkan tren total pesanan dari waktu ke waktu (tidak terpengaruh filter).")
        
        # Data untuk Tren (sesuai sel 87)
        orders_per_month_all = (
            orders_main_df.set_index('order_purchase_timestamp')
            .resample('M')['order_id']
            .count()
            .reset_index()
        )
        orders_per_month_all.rename(columns={'order_id': 'Jumlah Pesanan'}, inplace=True)
        
        # Plot Tren (sesuai sel 88)
        fig_trend, ax_trend = plt.subplots(figsize=(14, 6))
        ax_trend.plot(
            orders_per_month_all['order_purchase_timestamp'],
            orders_per_month_all['Jumlah Pesanan'],
            marker='o',
            color='#007ACC',
            linewidth=2,
            label='Jumlah Pesanan'
        )
        # Kustomisasi Sumbu X
        ax_trend.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_trend.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45, ha='right')
        
        # Garis Rata-rata
        rata_rata_all = orders_per_month_all['Jumlah Pesanan'].mean()
        ax_trend.axhline(y=rata_rata_all, color='#2C3E50', linestyle='--', label=f'Rata-rata: {rata_rata_all:,.0f}')
        
        ax_trend.set_title("Tren Volume Penjualan Bulanan (2016 - 2018)", loc='left', fontsize=16, fontweight='bold')
        ax_trend.set_xlabel("Periode Pembelian")
        ax_trend.set_ylabel("Jumlah Pesanan Unik")
        ax_trend.legend(loc='upper left')
        sns.despine()
        st.pyplot(fig_trend)


    # --- TAB 2: Analisis Pendapatan & Pesanan ---
    with tab_pendapatan:
        st.header("Analisis Pendapatan dan Pesanan")
        st.markdown("Data berdasarkan filter di *sidebar*.")

        # Pesanan per Kuartal (sesuai sel 85)
        st.subheader("Volume Pesanan per Kuartal")
        quarterly_orders = filtered_orders_df.groupby(['order_purchase_year', 'order_purchase_quarter'])['order_id'].nunique().reset_index()
        quarterly_orders['Kuartal'] = 'Q' + quarterly_orders['order_purchase_quarter'].astype(str)
        
        fig_q, ax_q = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x='Kuartal',
            y='order_id',
            hue='order_purchase_year',
            data=quarterly_orders,
            palette='viridis',
            ax=ax_q
        )
        ax_q.set_title("Perbandingan Volume Pesanan per Kuartal", fontsize=16)
        ax_q.set_ylabel("Jumlah Pesanan Unik")
        ax_q.set_xlabel("Kuartal Pembelian")
        st.pyplot(fig_q)

        # Pendapatan per Bulan (sesuai sel 81 - dimodifikasi untuk pendapatan)
        st.subheader("Pendapatan per Bulan")
        monthly_revenue = filtered_orders_df.set_index('order_purchase_timestamp').resample('M')['payment_value'].sum()
        
        fig_m, ax_m = plt.subplots(figsize=(12, 6))
        ax_m.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linestyle='-')
        ax_m.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_m.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_m.set_title("Tren Pendapatan Bulanan (Berdasarkan Filter)", fontsize=16)
        ax_m.set_ylabel("Total Pendapatan (Payment)")
        ax_m.set_xlabel("Bulan Pembelian")
        plt.xticks(rotation=45)
        st.pyplot(fig_m)

        # Menampilkan data dari pertanyaan bisnis 1 (Pendapatan Q4 2017)
        st.subheader("Studi Kasus: Pendapatan Q4 2017 (Sesuai Pertanyaan Bisnis)")
        # Menggunakan full_items_df untuk mereplikasi perhitungan 'price' dari notebook (sel 61)
        q4_2017_items = full_items_df[
            (full_items_df['order_purchase_year'] == 2017) &
            (full_items_df['order_purchase_quarter'] == 4)
        ]
        revenue_q4_2017_price = q4_2017_items['price'].sum()
        revenue_q4_2017_payment = q4_2017_items.drop_duplicates(subset=['order_id'])['payment_value'].sum()

        col_q4_1, col_q4_2 = st.columns(2)
        col_q4_1.metric("Pendapatan Q4 2017 (Total 'Price' Item)", f"R$ {revenue_q4_2017_price:,.2f}")
        col_q4_2.metric("Pendapatan Q4 2017 (Total 'Payment Value')", f"R$ {revenue_q4_2017_payment:,.2f}")
        st.caption("Notebook Anda (sel 61) menghitung `price.sum()`. Metrik `payment_value` mungkin lebih akurat untuk total pendapatan bersih.")


    # --- TAB 3: Analisis Produk ---
    with tab_produk:
        st.header("Analisis Kategori Produk")
        st.markdown("Data berdasarkan filter di *sidebar*.")

        # Slider untuk Top N Kategori
        top_n = st.slider("Tampilkan Top N Kategori", min_value=5, max_value=50, value=10)

        # Agregasi data (sesuai sel 82, tapi pakai English names)
        category_orders = filtered_items_df.groupby('product_category_name_english')['order_id'].nunique().reset_index()
        category_orders.rename(columns={'order_id': 'Jumlah Pesanan'}, inplace=True)
        category_orders_top_n = category_orders.sort_values(by='Jumlah Pesanan', ascending=False).head(top_n)

        # Plot Top N Kategori (sesuai sel 83)
        fig_cat, ax_cat = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x='Jumlah Pesanan',
            y='product_category_name_english',
            data=category_orders_top_n,
            palette='coolwarm',
            ax=ax_cat
        )
        ax_cat.set_title(f"Top {top_n} Kategori Produk dengan Pesanan Terbanyak", fontsize=16)
        ax_cat.set_xlabel("Jumlah Pesanan Unik")
        ax_cat.set_ylabel("Kategori Produk (Bahasa Inggris)")
        st.pyplot(fig_cat)

        # Opsi untuk melihat semua data kategori
        if st.checkbox("Tampilkan semua data kategori"):
            st.dataframe(category_orders.sort_values(by='Jumlah Pesanan', ascending=False))


    # --- TAB 4: Analisis Pelanggan ---
    with tab_pelanggan:
        st.header("Analisis Pelanggan")

        # Pertanyaan Bisnis 2: Repeat Buyers (sesuai sel 68 & 69)
        st.subheader("Analisis Pelanggan Berulang (Repeat Buyers)")
        
        # Menggunakan data lengkap (bukan data terfilter) sesuai logika notebook
        customer_order_count = orders_main_df.groupby('customer_unique_id')['order_id'].nunique().reset_index(name='order_count')
        repeat_customer_df = customer_order_count[customer_order_count['order_count'] > 1]
        
        repeat_customers_count = len(repeat_customer_df)
        total_customers_raw = len(customers_df) # Sesuai sel 69
        
        repeat_rate = (repeat_customers_count / total_customers_raw) * 100

        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Total Pelanggan Berulang (Order > 1)", f"{repeat_customers_count:,}")
        col_p2.metric("Persentase Pelanggan Berulang (dari Total Entri)", f"{repeat_rate:.2f}%")
        st.caption(f"Perhitungan sesuai notebook: {repeat_customers_count} / {total_customers_raw} (total entri customers_df).")
        
        st.markdown("---")

        # Wawasan dari Analisis Lanjutan (RFM & Retensi - sel 90-96)
        st.subheader("Segmentasi & Retensi (dari Analisis Lanjutan Q4 2017)")
        st.markdown("Wawasan ini didasarkan pada analisis RFM dan kohort retensi 90 hari yang dilakukan di *notebook* (sel 90-96).")

        st.subheader("Segmentasi RFM (Q4 2017)")
        col_rfm1, col_rfm2, col_rfm3 = st.columns(3)
        col_rfm1.metric("Champions (F=4, M=4)", "23")
        col_rfm2.metric("Loyal Customers (F=4, M=2/3)", "0")
        col_rfm3.metric("High-Value Spenders (F=2/3, M=4)", "189")
        
        st.subheader("Analisis Retensi 90 Hari (Kohort Q4 2017)")
        col_ret1, col_ret2 = st.columns(2)
        col_ret1.metric("Tingkat Retensi Pembeli Pertama (FTB)", "4.95%")
        col_ret2.metric("Tingkat Retensi Pelanggan Berulang (RC)", "16.47%")
        
        st.markdown(f"**Perbedaan:** Pelanggan yang sudah berulang **232.6%** lebih mungkin untuk membeli lagi dalam 90 hari dibandingkan pembeli pertama.")
        
        st.info("""
        **Insight dari Notebook (Sel 96):**
        1.  **Rintangan Retensi**: Rintangan utama bisnis adalah konversi dari pesanan pertama ke pesanan kedua (hanya 4.95% yang kembali).
        2.  **Potensi Tinggi**: Terdapat 189 "High-Value Spenders" (pembeli bernilai tinggi) yang merupakan target utama untuk dikonversi menjadi "Champions" (pembeli loyal bernilai tinggi).
        """)

else:
    st.warning("Gagal memuat data. Pastikan file CSV ada di folder 'data/'.")