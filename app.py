"""
Fashion Recommender Streamlit App - Product Detail View
Single product detail with recommendations and random prices
"""

import streamlit as st
import pandas as pd
import random
import hashlib
from pathlib import Path
from src.recommender import FashionRecommender
from PIL import Image
import re


def generate_consistent_price(product_id, base_min=29, base_max=149):
    """
    Generate consistent pricing for products using deterministic random generation.
    Ensures the same product_id always gets the same price.
    
    Args:
        product_id: Unique product identifier
        base_min: Minimum price range
        base_max: Maximum price range
    
    Returns:
        Consistent price for the product
    """
    # Create a hash of the product_id for deterministic randomness
    hash_obj = hashlib.md5(str(product_id).encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    
    # Use the hash as seed for consistent randomness
    rng = random.Random(seed)
    return rng.randint(base_min, base_max)


def generate_similar_price(base_price, deviation_percent=5):
    """
    Generate a price similar to base price with controlled deviation (±5%).
    
    Args:
        base_price: Base price to vary from
        deviation_percent: Maximum percentage deviation (default 5%)
    
    Returns:
        Price within ±deviation_percent of base_price
    """
    deviation = base_price * (deviation_percent / 100)
    min_price = max(1, base_price - deviation)  # Ensure price doesn't go below 1
    max_price = base_price + deviation
    
    # Use multiple random factors for better variation
    import time
    import os
    seed = int((time.time() * 1000000) % 10000) + os.getpid()
    rng = random.Random(seed)
    
    return round(rng.uniform(min_price, max_price))


def get_front_facing_image_path(image_path):
    """
    Convert image path to front-facing version if it exists.
    
    Args:
        image_path: Original image path (e.g., "img/MEN/Denim/id_00000080/02_1_front.jpg")
    
    Returns:
        Path to front-facing image if exists, otherwise None
    """
    if pd.isna(image_path):
        return None
    
    path_str = str(image_path)
    
    # Check if it already contains "_front.jpg" - if so, use it as is
    if "_front.jpg" in path_str:
        full_path = Path("Data") / path_str
        if full_path.exists():
            return path_str
        return None
    
    # Otherwise, look for any _front.jpg variant in the same product directory
    path_obj = Path(path_str)
    product_dir = path_obj.parent
    full_product_dir = Path("Data") / product_dir
    
    # Search for any file ending in _front.jpg in this directory
    if full_product_dir.exists():
        front_files = list(full_product_dir.glob("*_front.jpg"))
        if front_files:
            # Return the first front-facing image found
            return str(product_dir / front_files[0].name)
    
    return None

# Page config
st.set_page_config(
    page_title="StyleMatch - Product Detail",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit header and top bar */
    header {visibility: hidden;}
    
    /* Override Streamlit's main container background */
    .main {
        background: #f8f9fa !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #f8f9fa !important;
    }
    
    body {
        background: #f8f9fa !important;
        min-height: 100vh;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .brand-header {
        text-align: center;
        padding: 20px 24px 8px 24px;
        background: #ffffff;
        margin-bottom: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        border-bottom: 1px solid #e0e0e0;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .brand-logo {
        font-size: 48px;
        font-weight: 900;
        color: #1f2937;
        margin: 0;
    }
    
    .brand-name {
        font-size: 48px;
        font-weight: 900;
        color: #1f2937;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .brand-tagline {
        font-size: 12px;
        color: #999;
        margin: 4px 0 0 0;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .brand-info {
        text-align: center;
    }
    
    .filter-chip {
        display: inline-block;
        padding: 8px 16px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
        margin-right: 8px;
    }
    
    .filter-chip.active {
        background: #1f2937;
        color: white;
        border-color: #1f2937;
    }
    
    .size-button {
        padding: 10px 14px;
        border: 1.5px solid #d1d5db;
        border-radius: 6px;
        background: white;
        cursor: pointer;
        font-size: 13px;
        font-weight: 600;
        transition: all 0.2s;
        margin-right: 8px;
    }
    
    .size-button.active {
        background: #000;
        color: white;
        border-color: #000;
    }
    
    .size-button.disabled {
        background: #f3f4f6;
        color: #9ca3af;
        cursor: not-allowed;
        border-color: #e5e7eb;
    }
    
    .cta-button {
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
    }
    
    .cta-button.primary {
        background: #1f2937;
        color: white;
    }
    
    .cta-button.primary:hover {
        background: #000;
    }
    
    .cta-button.primary:disabled {
        background: #d1d5db;
        cursor: not-allowed;
    }
    
    .cta-button.secondary {
        background: white;
        color: #1f2937;
        border: 1.5px solid #d1d5db;
    }
    
    .cta-button.secondary:hover {
        border-color: #1f2937;
    }
    
    .recommendations-carousel {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 24px;
        scroll-snap-type: x mandatory;
        -webkit-overflow-scrolling: touch;
    }
    
    .rec-card {
        flex: 0 0 160px;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        scroll-snap-align: start;
        transition: all 0.2s;
    }
    
    .rec-card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
        transform: translateY(-4px);
    }
    
    .rec-image {
        width: 100%;
        aspect-ratio: 3/4;
        object-fit: cover;
    }
    
    .rec-info {
        padding: 12px;
    }
    
    .rec-price {
        font-weight: 700;
        font-size: 14px;
        color: #000;
        margin-bottom: 4px;
    }
    
    .rec-tag {
        font-size: 11px;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Product title and price - updated hierarchy */
    .product-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 16px;
        line-height: 1.2;
    }
    
    .product-price {
        font-size: 28px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 20px;
    }
    
    .stock-badge {
        display: inline-block;
        padding: 8px 12px;
        background: #dc2626;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .product-meta {
        margin: 24px 0;
        padding: 16px 0;
        border-top: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .meta-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        font-size: 13px;
    }
    
    .meta-label {
        color: #6b7280;
        font-weight: 600;
    }
    
    .meta-value {
        color: #1f2937;
    }
    
    .action-buttons {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 24px 0;
    }
    
    .recommendations-section {
        margin-top: 40px;
    }
    
    .recommendations-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .recommendation-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .filter-section {
        background: transparent;
        padding: 8px 24px;
        border-radius: 0;
        margin-bottom: 0;
        border-bottom: none;
    }
    
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    
    .modal-content {
        background: white;
        border-radius: 12px;
        padding: 30px;
        max-width: 90%;
        max-height: 90vh;
        overflow-y: auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .modal-header {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .modal-close {
        font-size: 28px;
        cursor: pointer;
        color: #999;
    }
    
    .modal-close:hover {
        color: #1f2937;
    }
    
    .side-modal {
        position: fixed;
        right: 0;
        top: 0;
        width: 450px;
        height: 100vh;
        background: white;
        box-shadow: -4px 0 20px rgba(0,0,0,0.15);
        overflow-y: auto;
        z-index: 999;
        padding: 30px;
    }
    
    .side-modal-header {
        font-size: 22px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .side-close {
        font-size: 24px;
        cursor: pointer;
        color: #999;
    }
    
    .side-close:hover {
        color: #1f2937;
    }
    
    .side-item {
        padding: 15px;
        border-bottom: 1px solid #e5e7eb;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .side-item:hover {
        background: #f9fafb;
    }
    
    .side-item-name {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 5px;
    }
    
    .side-item-price {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 5px;
    }
    
    .side-item-similarity {
        font-size: 12px;
        color: #10b981;
        font-weight: 600;
    }
    
    .side-rec-button {
        background: #0056b3;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 600;
        font-size: 12px;
        transition: all 0.3s;
    }
    
    .side-rec-button:hover {
        background: #003d82;
    }
</style>
""", unsafe_allow_html=True)

# Initialize recommender
@st.cache_resource
def load_recommender():
    """Load the recommender engine"""
    return FashionRecommender(index_dir="./Index", verbose=False)

recommender = load_recommender()

# Get product data
products_df = recommender.products.copy()

# Extract gender from image path if not in dataframe
if 'gender' not in products_df.columns:
    products_df['gender'] = products_df['image_path'].str.extract(r'(WOMEN|MEN)', expand=False)

# Filter to FRONT-FACING products only
from src.sim_helpers import filter_to_front_facing
products_df = filter_to_front_facing(products_df)

# Add consistent random prices and stock status
if 'price' not in products_df.columns or products_df['price'].isna().any():
    # Generate consistent prices based on product_id for reproducibility
    products_df['price'] = products_df['product_id'].apply(generate_consistent_price)

if 'is_in_stock' not in products_df.columns:
    # Use product_id for consistent stock status
    products_df['is_in_stock'] = products_df['product_id'].apply(
        lambda pid: 1 if generate_consistent_price(pid, 0, 100) > 20 else 0
    )

# Initialize session state
if 'current_product_id' not in st.session_state:
    st.session_state.current_product_id = None
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False

# Get unique values
genders = sorted(products_df['gender'].unique())

# Display brand header
st.markdown("""
<div class="brand-header">
    <div class="brand-info">
        <div style="position: relative; display: inline-block;">
            <span style="position: absolute; top: -8px; left: 0; font-size: 36px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1)); transform: rotate(-15deg);">👑</span>
            <h1 class="brand-name">VERA</h1>
        </div>
        <p class="brand-tagline">DISCOVER YOUR STYLE</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Filter section
st.markdown('<div class="filter-section">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 1.5, 1.2])

with col1:
    selected_gender = st.selectbox("👥 Gender", genders, key="gender_filter")

with col2:
    gender_products = products_df[products_df['gender'] == selected_gender]
    available_categories = sorted(gender_products['category'].unique())

    category_options = ["All"] + available_categories
    selected_category = st.selectbox("👗 Category", category_options, key="category_filter")

with col3:
    st.write("")
    st.write("")
    if st.button("🎲 Random Product", use_container_width=True):
        if selected_category == "All":
            filtered_products = gender_products
        else:
            filtered_products = gender_products[gender_products['category'] == selected_category]

        if len(filtered_products) > 0:
            random_product = filtered_products.sample(1).iloc[0]
            st.session_state.current_product_id = random_product['product_id']
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Get current product or pick first in filtered list
if selected_category == "All":
    filtered_products = gender_products
else:
    filtered_products = gender_products[gender_products['category'] == selected_category]


# 2. Initialize current_product_id if None and we have filtered products
if st.session_state.current_product_id is None and len(filtered_products) > 0:
    st.session_state.current_product_id = filtered_products.iloc[0]['product_id']

# 3. If current_product_id is set but not in the current filtered subset,
#    reset it to the first product in filtered_products (if any).
if (
    st.session_state.current_product_id is not None
    and len(filtered_products) > 0
    and st.session_state.current_product_id not in filtered_products['product_id'].values
):
    st.session_state.current_product_id = filtered_products.iloc[0]['product_id']

if st.session_state.current_product_id is not None and len(filtered_products) > 0:
    # 4. Get product details from the currently filtered subset
    current_rows = filtered_products[filtered_products['product_id'] == st.session_state.current_product_id]
    if current_rows.empty:
        # Fallback safety: default to first filtered product
        current_rows = filtered_products.iloc[[0]]
        st.session_state.current_product_id = current_rows.iloc[0]['product_id']
    product = current_rows.iloc[0]
    # ⚠️ 原来这里有一行 product_info = recommender.get_product_info(...)，
    # 我们删除它，因为当前 FashionRecommender 没有这个方法，也没有被用到。
    
    # Main product detail
    st.markdown('<div class="product-detail-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.5, 1.2])
    
    with col1:
        # Product image - use front-facing if available
        front_facing_path = get_front_facing_image_path(product['image_path'])
        if front_facing_path:
            product_image_path = Path("Data") / front_facing_path
        else:
            product_image_path = Path("Data") / product['image_path']
        
        if product_image_path.exists():
            img = Image.open(product_image_path)
            st.image(img, use_container_width=True)
        else:
            st.info("Image not found")
    
    with col2:
        # Product details
        st.markdown(f"<div class='product-title'>{product['category']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='product-price' style='color: #16a34a; font-size: 3em;'>${product['price']:.0f}</div>", unsafe_allow_html=True)
        
        # Out of stock badge (without X symbol)
        st.markdown("<div class='stock-badge'>OUT OF STOCK</div>", unsafe_allow_html=True)
        
        # Randomly select one size
        sizes = ['XS', 'S', 'M', 'L', 'XL']
        selected_size = random.choice(sizes)  # Only one size is selected
        
        # Build all size buttons in a single HTML string
        buttons_html = "<div style='display: flex; gap: 8px; flex-wrap: nowrap;'>"
        for size in sizes:
            if size == selected_size:
                buttons_html += f"<button style='padding: 8px 12px; border: 2px solid #333; border-radius: 4px; background: #333; color: white; cursor: pointer; font-size: 12px; font-weight: 600;'>{size}</button>"
            else:
                buttons_html += f"<button style='padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; font-size: 12px;'>{size}</button>"
        buttons_html += "</div>"
        
        st.markdown(f"""
        <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid #eee; margin-bottom: 20px;'>
            <div style='margin-bottom: 12px;'>
                <span style='color: #666; font-size: 12px; font-weight: 600;'>Size:</span>
            </div>
            {buttons_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("🛒 Add to Bag", use_container_width=True, type="primary", key="add_bag")
        with col_btn2:
            st.button("❤️ Wishlist", use_container_width=True, key="add_wish")
        
        # Get recommendations
        try:
            filter_category_flag = (selected_category != "All")

            recommendations = recommender.recommend_similar_products(
              product_id=st.session_state.current_product_id,
              top_k=40,
              filter_category=filter_category_flag,
              filter_in_stock=False,
              return_similarity=True
)

            
            # Apply similarity threshold filter (hardcoded to 0.50)
            recommendations = recommendations[recommendations['similarity'] >= 0.50]
            
            # Filter out the current product
            current_product_id = st.session_state.current_product_id
            recommendations = recommendations[recommendations['product_id'] != current_product_id]
            
            # Remove any duplicates
            recommendations = recommendations.drop_duplicates(subset=['product_id'], keep='first')
            
            # Get the main product's front-facing image path for comparison
            main_front_facing_path = get_front_facing_image_path(product['image_path'])
            if main_front_facing_path:
                main_display_path = main_front_facing_path
            else:
                main_display_path = product['image_path']
            
            # Filter to only recommendations with front-facing images and unique image paths
            front_facing_recs = []
            seen_image_paths = set()
            seen_image_paths.add(main_display_path)  # Track main image to avoid duplicates
            
            # Get main product's gender
            main_product_gender = product['gender']
            
            for _, rec in recommendations.iterrows():
                rec_product = products_df[products_df['product_id'] == rec['product_id']]
                if len(rec_product) > 0:
                    # Check if recommendation matches main product's gender
                    rec_gender = rec_product.iloc[0]['gender']
                    if rec_gender != main_product_gender:
                        continue
                    
                    rec_image_path = rec_product.iloc[0]['image_path']
                    # Check if front-facing image exists
                    rec_front_facing_path = get_front_facing_image_path(rec_image_path)
                    if rec_front_facing_path:
                        # Only add if image path hasn't been seen yet
                        if rec_front_facing_path not in seen_image_paths:
                            front_facing_recs.append(rec)
                            seen_image_paths.add(rec_front_facing_path)
            
            if front_facing_recs:
                recommendations = pd.DataFrame(front_facing_recs)
            else:
                recommendations = pd.DataFrame()
            
            # Limit to max 5 recommendations (all unique at this point)
            recommendations = recommendations.head(5)
            
        except Exception:
            recommendations = pd.DataFrame()
        
        if len(recommendations) > 0:
            # Similar Items bubble inside col2 (top left area)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.container():
                st.markdown("""
                <h3 style='
                    color: #333;
                    margin: 0 0 20px 0;
                    font-size: 18px;
                    font-weight: 600;
                '>Items you might like</h3>
                """, unsafe_allow_html=True)
                
                # Display items in a grid (1-5 columns based on count)
                num_cols = min(len(recommendations), 5)
                cols = st.columns(num_cols)
                
                for idx, (_, rec) in enumerate(recommendations.iterrows()):
                    with cols[idx]:
                        rec_price = None
                        rec_image_path = None
                        
                        # Get price and image from products_df
                        rec_product = products_df[products_df['product_id'] == rec['product_id']]
                        if len(rec_product) > 0:
                            base_rec_price = rec_product.iloc[0]['price']
                            # Generate similar price within ±5% of current product
                            rec_price = generate_similar_price(product['price'], deviation_percent=5)
                            # Get front-facing image path
                            original_path = rec_product.iloc[0]['image_path']
                            front_facing_path = get_front_facing_image_path(original_path)
                            if front_facing_path:
                                rec_image_path = Path("Data") / front_facing_path
                            else:
                                rec_image_path = Path("Data") / original_path
                        else:
                            # Fallback: generate similar price to current product
                            rec_price = generate_similar_price(product['price'], deviation_percent=5)
                        
                        # Display image if found
                        if rec_image_path and rec_image_path.exists():
                            try:
                                img = Image.open(rec_image_path)
                                st.image(img, use_container_width=True)
                            except Exception:
                                pass
                        
                        # Product info
                        st.markdown(f"""
                        <div style='text-align: center; margin-top: 6px;'>
                            <div style='font-weight: 600; color: #333; font-size: 10px; line-height: 1.2; margin-bottom: 4px;'>{rec['category']}</div>
                            <div style='font-size: 20px; font-weight: 700; color: #16a34a;'>${rec_price:.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                
                st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            num_cols_btns = min(len(recommendations), 5)
            cols_btns = st.columns(num_cols_btns)
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                with cols_btns[idx]:
                    if st.button("View", key=f"rec_view_{rec['product_id']}", use_container_width=True):
                        st.session_state.current_product_id = rec['product_id']
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

st.divider()
