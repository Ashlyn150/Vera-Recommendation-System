"""
Simulation Page
A/B test simulation to evaluate recommendation policies
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import hashlib
import random
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sim_config import get_personas, get_default_mix, get_persona_params, create_custom_params, get_persona_table_data
from src.sim_helpers import get_products_df, pick_oos_factory, baseline_factory, create_recommend_wrapper, filter_to_front_facing
from src.simulator import simulate_sessions, simulate_ab_test, kpis
# ⭐ 改成从 KNN 推荐器里加载
from src.recommender_knn import FashionRecommenderKNN

# Page config
st.set_page_config(
    page_title="Simulation",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for simulator page
st.markdown("""
<style>
    .main {
        background: #f8f9fa !important;
    }
    
    .simulator-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .formula-box {
        background: #f0f4f8;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header —— 文案从 ANN 改成 KNN
st.markdown("""
<div class="simulator-header">
    <h1>🔍 KNN Recommender Simulator</h1>
    <p>Predict real-world impact of K-Nearest Neighbors recommendations on revenue and engagement</p>
</div>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def get_recommender():
    """Load KNN recommender instance"""
    # ⭐ 使用 KNN 版的 FashionRecommenderKNN
    return FashionRecommenderKNN(index_dir="./Index", verbose=False)

recommender = get_recommender()

# Get products data (read-only from app)
@st.cache_data
def load_products():
    """Load products DataFrame"""
    return get_products_df(st.session_state)

# 检查-之后需要再修改----------------------------------------------------------------------------
# ==========================================
# 直接从 KNN 推荐器里拿产品表，绕开 load_products()
# ==========================================

# 看看 recommender 本身有多少行
#st.write("🔍 [DEBUG] recommender.products rows:", len(recommender.products))

products_df = recommender.products.copy()

# 如果真的还是空的，直接报错提示你，不再继续往下跑
if products_df is None or len(products_df) == 0:
    st.error("❌ products_df is empty – 请检查 Index/products.pkl 是否正常！")
else:
    # 补充 gender 列
    if 'gender' not in products_df.columns:
        products_df['gender'] = products_df['image_path'].str.extract(r'(WOMEN|MEN)', expand=False)

    # 只留正面图片（和你队友 ANN 一样）
    products_df = filter_to_front_facing(products_df)

# 再打印一次最终可用的 products_df 信息
#st.write("🔍 [DEBUG] products_df rows (after filter):", len(products_df))
#st.write("🔍 [DEBUG] products_df columns:", list(products_df.columns))
#st.write("🔍 [DEBUG] recommender.products rows:", len(recommender.products))
#st.write("🔍 [DEBUG] products_df rows (after filter):", len(products_df))


#----------------------------------------------------------------------------

# Ensure prices and stock are set
def generate_consistent_price(product_id, base_min=29, base_max=149):
    hash_obj = hashlib.md5(str(product_id).encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    rng = random.Random(seed)
    return rng.randint(base_min, base_max)

if 'price' not in products_df.columns:
    products_df['price'] = products_df['product_id'].apply(generate_consistent_price)

if 'is_in_stock' not in products_df.columns:
    products_df['is_in_stock'] = products_df['product_id'].apply(
        lambda pid: 1 if generate_consistent_price(pid, 0, 100) > 20 else 0
    )

# ============================================================================
# Sources & Assumptions Rendering Function
# ============================================================================

def render_sources_and_assumptions():
    """
    Render academic sources and citations for the simulator.
    ⚠️ 这里仍然是你队友针对 ANN 写的参考文献，如果你有时间/心情，
       可以把里面 "ANN" 的字眼改成 "KNN"，
       不改也不影响代码运行，只是文案有点不对应。
    """
    with st.expander("📚 Academic Sources", expanded=False):
        st.markdown("""
        ### Verified Academic Sources
        
        **1. A/B Testing & Causal Inference** 🧪
        
        Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
        - **ISBN:** 978-1108724265
        - **Link:** https://www.cambridge.org/9781108724265
        - **How it shaped the simulator:** Provides statistical foundations for A/B testing framework, including proper randomization, metric selection, and result interpretation.
        - **Endorsement:** Recommended by Google, Microsoft, LinkedIn, and Harvard Business School
        
        ---
        
        **2. Out-of-Stock Bounce Rate: 65% Baseline** 📊
        
        **Primary source:**
        Fitzsimons, G. J. (2000). Consumer response to stockouts. *Journal of Consumer Research*, 27(2), 249-266.
        - **DOI:** 10.1086/314320
        - **Link:** https://academic.oup.com/jcr/article-abstract/27/2/249/1786037
        - **Finding:** Customers bounce when products are unavailable; will purchase alternatives if presented well
        - **How it's used:** Foundation for our quality-driven bounce mechanism—better recommendations lead to naturally lower bounce rates
        
        **Recent validation:**
        - Hoang, N., & Breugelmans, E. (2023). Substitution policies and the dominant attribute in choice behavior. *Journal of Retailing*, 99(3), 341-357. **DOI:** 10.1016/j.jretai.2023.04.005
          - Finding: 60-70% of consumers switch to competitor or abandon purchase entirely at OOS
          - Validates: 65% bounce rate is realistic for random/low-quality recommendations
        
        - Breugelmans, E., & Campo, K. (2011). Effectiveness of in-store displays in a virtual store environment. *Journal of Retailing*, 87(1), 75-89. **DOI:** 10.1016/j.jretai.2010.11.006
          - Finding: Without good alternatives, OOS abandonment is the most common response
          - Validates: High baseline bounce rate when recommendations lack quality
        
        ---
        
        **3. Improved Bounce Rate with Good Recommendations: 40%** ✅
        
        Hoang, N., & Breugelmans, E. (2023) — same as above
        - Finding: When good alternatives are presented, bounce rate drops to 35-45%
        - Validates: 40% model bounce rate reflects strong recommendation quality
        
        ---
        
        **4. Click-Through Rate: 25% Baseline** 🖱️
        
        Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of recommender algorithms on top-N recommendation tasks. *Proceedings of the 4th ACM Conference on Recommender Systems*. **DOI:** 10.1145/1864708.1864721
        - Finding: 20-30% CTR is typical for random/baseline recommendations in e-commerce
        - Validates: 25% baseline CTR is industry-standard
        
        He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., ... & Chang, Y. (2014). Practical lessons from predicting clicks on ads at Facebook. *Proceedings of the 20th ACM SIGKDD International Conference*. **DOI:** 10.1145/2648584.2648589
        - Finding: CTR across major platforms averages 2-5% for untargeted; 20-40% for targeted content
        - Validates: 25% is realistic for random product recommendations
        
        ---
        
        **5. Click Uplift with Better Recs: 1.5x (50% improvement)** 🚀
        
        Ekstrand, M. D., Harper, F. M., Willemsen, M. C., & Konstan, J. A. (2014). User perception of differences in recommender algorithms. *Proceedings of the 8th ACM Conference on Recommender Systems*. **DOI:** 10.1145/2645710.2645737
        - Finding: Personalized recommendations outperform baseline by 40-100% in click-through
        - Validates: 1.5x uplift is realistic vs random baseline
        
        ---
        
        **6. Purchase Conversion Rate: 10% of Clickers** 💳
        
        Industry standard: E-commerce conversion rates typically range from 2-5% for general sites to 10-15% for targeted fashion retail. Our 10% figure represents mid-range expectations for quality recommendations in fashion e-commerce.
        
        ---
        
        ### How These Sources Informed Our Simulator
        
        - **Bounce behavior** (Fitzsimons + Hoang/Breugelmans + Breugelmans/Campo): Quality emerges from availability and recommendation effectiveness
        - **Baseline CTR** (Cremonesi + He et al.): 25% reflects industry standards for undifferentiated recommendations
        - **Model uplift** (Ekstrand): 1.5x improvement is achievable with better matching
        - **Conversion funnel**: 10% aligns with realistic e-commerce expectations for fashion retail
        
        Together, these peer-reviewed sources support our core innovation: bounce rates emerging naturally from recommendation quality rather than artificial slider adjustments.
        """)


st.sidebar.markdown("---")

# Experiment Mode Selector
st.sidebar.markdown("### 🎯 Deploy Your KNN Recommendation Engine")

# A/B Test mode is the only option
experiment_mode = "A/B Test (Current vs KNN)"

# Keep mode and seed in session state for use later
# ⭐ 这里把模式改成 'knn'
mode = 'knn'  # KNN is the current engine mode

# Provide simple session count and seed in a collapsible section
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    n_sessions = st.slider(
        "Number of Simulated Customers",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000,
        help="More customers = more statistically significant results. 5,000-10,000 is typically sufficient. Higher values take longer to compute."
    )
    
    seed = st.number_input(
        "Randomization Seed",
        min_value=0,
        max_value=999999,
        value=42,
        help="Use same seed to reproduce results (A/B test consistency)"
    )

# Use defaults if not set via expander
if 'n_sessions' not in locals():
    n_sessions = 1000
if 'seed' not in locals():
    seed = 42

price_band = 20  # Fixed value for now

st.sidebar.markdown("---")

# Create final params (use default persona)
default_persona = get_default_mix()
sim_params = get_persona_params(default_persona)

# ============================================================================
# A/B TEST: CURRENT VS KNN
# ============================================================================

st.sidebar.markdown("### 🔄 A/B Test Configuration")

# Traffic Split
col_split1, col_split2 = st.sidebar.columns([5, 1])
with col_split1:
    ab_split = st.slider(
        "% of traffic to KNN",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help="Split test traffic between current system (control) and KNN system (treatment). Example: 50% means half the customers see each system."
    )
st.sidebar.caption(f"📊 {100-ab_split}% current | {ab_split}% KNN")

st.sidebar.markdown("---")

# Toggle between dynamic and fixed bounce
st.sidebar.markdown("### 🎯 Bounce Rate Control")
use_dynamic_bounce = st.sidebar.checkbox(
    "Use product-quality-based bounce rates (Recommended)",
    value=False,
    help="""
    ✅ Recommended: Bounce rates emerge from recommendation quality (realistic)
    ❌ Legacy: Fixed bounce rates from sliders (scenario testing)
    
    **Difference:**
    - Recommended: Better recommendations naturally keep more customers
    - Legacy: You set bounce rates directly with sliders (artificial)
    """
)

st.sidebar.markdown("---")

# Hardcoded baseline parameters based on academic research
# These apply to Current System only and are NOT adjustable
BASELINE_OOS_BOUNCE = 0.65  # Fitzsimons (2000): 65% baseline bounce at OOS
BASELINE_CLICK_RATE = 0.25  # Cremonesi et al. (2010): 25% CTR for random recs
BASELINE_PURCHASE_RATE = 0.10  # Industry standard: 10% conversion for fashion

if use_dynamic_bounce:
    st.sidebar.markdown("#### KNN Model Parameters")
    st.sidebar.caption("These parameters control KNN system performance only")
    
    # User Sensitivity (KNN-specific)
    quality_sensitivity = st.sidebar.slider(
        "User Sensitivity (KNN)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help="""
        How picky are KNN customers about recommendation quality?
        
        **0.0**: Not picky (KNN customers bounce regardless of quality)
        **0.5**: Moderately picky (default, quality matters somewhat)
        **1.0**: Very picky (quality strongly affects whether they stay)
        
        Note: Baseline customers use fixed 65% bounce (academic standard).
        """
    )
    st.sidebar.caption("How much KNN customers care about recommendation quality")
    
    st.sidebar.markdown("---")
    
    # Baseline OOS Bounce Rate (KNN-specific)
    col_base1, col_base2 = st.sidebar.columns([5, 1])
    with col_base1:
        baseline_oos_bounce_knn = st.slider(
            "KNN Base Bounce Rate",
            min_value=0,
            max_value=100,
            value=65,
            step=5,
            format="%d%%",
            key="quality_baseline_bounce_slider",
            help="""
            Starting bounce % for KNN customers before quality adjustment.
            
            **20%**: Optimistic (most KNN customers stay even without good recs)
            **65%**: Realistic baseline (industry standard)
            **95%**: Pessimistic (almost everyone leaves)
            
            Note: Baseline system uses fixed 65% (Fitzsimons 2000).
            """
        ) / 100.0
    st.sidebar.caption("Starting bounce % for KNN customers only")
    
    # KNN Accuracy
    similarity_threshold = st.sidebar.slider(
        "KNN Accuracy",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        format="%.2f",
        help="""
        How accurate are KNN recommendations?
        
        **0.3**: Lenient (many products qualify)
        **0.6**: Balanced (default, recommended)
        **0.9**: Strict (only best matches qualify)
        
        Note: Baseline uses random recommendations (no accuracy threshold).
        """
    )
    st.sidebar.caption("Minimum similarity threshold for KNN recommendations")
    
    # Price Tolerance
    price_tolerance = st.sidebar.slider(
        "Price Tolerance",
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.05,
        format="±%.0f%%",
        help="""
        How much can recommended price differ from OOS price?
        
        **±10%**: Strict (customers want similar price)
        **±20%**: Moderate (default, realistic)
        **±50%**: Loose (price doesn't matter much)
        
        Note: Applies to both Baseline and KNN systems.
        """
    )
    st.sidebar.caption("Price range tolerance for all recommendations")
    
    # Hidden: Set baseline bounce to academic standard
    baseline_oos_bounce = BASELINE_OOS_BOUNCE
    engine_oos_bounce = 0.40
    
else:
    st.sidebar.markdown("#### Fixed Bounce Rates (Legacy Mode)")
    st.sidebar.warning(
        "You're using fixed bounce rates (legacy mode). For realistic predictions, switch to quality-based mode above."
    )
    
    st.sidebar.info(
        """
        **Baseline System (Current):**
        - Bounce Rate: 65% (fixed, Fitzsimons 2000)
        - Average Click Rate: ~25% (Cremonesi et al. 2010)
        - Average Purchase Rate: ~10% (industry standard)
        
        Note: Click/purchase rates vary by product quality but average to these academic benchmarks.
        """
    )
    
    # Baseline Bounce Rate (for legacy reference only)
    baseline_oos_bounce = BASELINE_OOS_BOUNCE
    
    # KNN Bounce Rate
    col_knn1, col_knn2 = st.sidebar.columns([5, 1])
    with col_knn1:
        engine_oos_bounce = st.slider(
            "KNN system bounce %",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            key="engine_bounce_slider",
            help="% who abandon at out-of-stock with KNN recommendations. Lower bounce = more viewers = more purchases = higher revenue/session."
        ) / 100.0
    st.sidebar.caption("% of KNN customers who leave without viewing recommendations")
    
    # Set default quality parameters (unused in legacy mode)
    quality_sensitivity = 0.5
    similarity_threshold = 0.6
    price_tolerance = 0.2

st.sidebar.markdown("---")

# Click Uplift (KNN only)
col_click1, col_click2 = st.sidebar.columns([5, 1])
with col_click1:
    engine_click_uplift = st.slider(
        "KNN click uplift",
        min_value=0.8,
        max_value=3.0,
        value=1.5,
        step=0.1,
        format="%.1fx",
        key="click_uplift_slider",
        help="How many times more viewers click with KNN vs baseline. 1.5x = 50% more clicks. Baseline averages ~25% (Cremonesi 2010), varies by product quality."
    )
st.sidebar.caption("Multiplier on baseline click rate for KNN recommendations")

st.sidebar.markdown("---")

# Purchase Uplift (KNN only)
col_purch1, col_purch2 = st.sidebar.columns([5, 1])
with col_purch1:
    engine_purchase_uplift = st.slider(
        "KNN purchase uplift",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        format="%.1fx",
        key="purchase_uplift_slider",
        help="Of clickers, how many times more likely to buy with KNN vs baseline. 1.0x typical. Baseline averages ~10% (industry standard), varies by product quality."
    )
st.sidebar.caption("Multiplier on baseline purchase rate for KNN recommendations")

st.sidebar.markdown("---")

# Average Purchase Price (applies to both systems)
avg_purchase_price = st.sidebar.slider(
    "Average purchase price",
    min_value=20,
    max_value=150,
    value=75,
    step=5,
    format="$%d",
    help="Average product price when customers purchase. Both systems use this same price. Range reflects product catalog ($20-$150)."
)

st.sidebar.markdown("---")

# Main area
st.markdown("## 🔍 KNN Recommender - A/B Test Simulator")

st.markdown("""
### Real-World Problem

Popular products sell out. Without good alternatives shown, **customers leave your store** (bounce).

### The Solution: K-Nearest Neighbors (KNN)

Instead of random suggestions, KNN intelligently finds **semantically similar, in-stock products** that match what customers were looking for.

**This simulator predicts:**
- 💰 **Revenue impact** when KNN recovers lost customers
- 📈 **Engagement improvement** (clicks, conversions) from better recommendations  
- 🎯 **Customer retention** when facing out-of-stock situations
- 📊 **Confidence levels** based on test traffic and volume
""")

# ------- 从这里往下的逻辑全部保持和原 ANN 版本一致 -------


# Run simulation button
st.markdown("---")
st.markdown("## 🚀 Run Simulation")

if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
    with st.spinner("Simulating user sessions..."):
        # Create helper functions
        print("[DEBUG] products_df rows in Simulation.py:", len(products_df))
        pick_oos_fn = pick_oos_factory(products_df)
        baseline_fn = baseline_factory(products_df)
        recommend_fn = create_recommend_wrapper(recommender, products_df)
        
        # Run A/B test simulation
        status_text = st.empty()
        status_text.text(f"Simulating {n_sessions:,} customer sessions...")
        progress_bar = st.progress(0.0)
        
        combined_logs = simulate_ab_test(
            n_sessions=n_sessions,
            mode=mode,
            price_band=price_band,
            seed=seed,
            params=sim_params,
            pick_oos_fn=pick_oos_fn,
            baseline_fn=baseline_fn,
            recommend_fn=recommend_fn,
            ab_split=ab_split / 100.0,
            baseline_oos_bounce=baseline_oos_bounce,
            engine_oos_bounce=engine_oos_bounce,
            engine_click_uplift=engine_click_uplift,
            engine_purchase_uplift=engine_purchase_uplift,
            baseline_avg_price=avg_purchase_price,
            engine_avg_price=avg_purchase_price,
            quality_sensitivity=quality_sensitivity,
            similarity_threshold=similarity_threshold,
            price_tolerance=price_tolerance,
            use_dynamic_bounce=use_dynamic_bounce
        )
        
        progress_bar.progress(1.0)
        status_text.text("Processing results...")
        
        # Calculate KPIs
        kpis_df = kpis(combined_logs)
        
        # Store in session state
        st.session_state.sim_logs = combined_logs
        st.session_state.sim_kpis = kpis_df
        
        status_text.empty()
        st.success(f"✅ A/B Test complete! Processed {n_sessions:,} sessions with {ab_split}% traffic to KNN system.")

# Display results if available
if 'sim_kpis' in st.session_state and st.session_state.sim_kpis is not None:
    st.markdown("---")
    st.markdown("## 📊 KNN Deployment Results")
    
    kpis_df = st.session_state.sim_kpis
    is_ab_test = True  # Always true now
    
    # KPI Table
    st.markdown("### 📊 Business Metrics Summary")
    st.markdown("*Comparing current system vs KNN recommendations for out-of-stock scenarios*")
    
    # Get logs to calculate detailed metrics
    logs_df = st.session_state.sim_logs
    
    # Prepare detailed metrics for both systems
    detailed_metrics = []
    
    for policy in ['baseline', 'engine']:
        policy_logs = logs_df[logs_df['policy'] == policy]
        n_sessions = len(policy_logs)
        
        # Calculate counts
        n_bounced = (policy_logs['bounced'] == True).sum()
        n_viewers = (policy_logs['bounced'] == False).sum()
        n_clickers = (policy_logs['clicked_rec'] == True).sum()
        n_purchasers = (policy_logs['purchased'] == True).sum()
        
        # Get metrics from KPI row
        kpi_row = kpis_df[kpis_df['policy'] == policy].iloc[0]
        
        # Calculate revenue per purchaser (only for those who actually bought)
        total_revenue = policy_logs['revenue'].sum()
        revenue_per_purchaser = (total_revenue / n_purchasers) if n_purchasers > 0 else 0
        
        detailed_metrics.append({
            'System': 'Current System' if policy == 'baseline' else 'KNN System',
            'Total Sessions': f"{n_sessions:,}",
            'Bounce Rate': f"{kpi_row['oos_bounce_pct']:.2f}%",
            'Bounced': f"{n_bounced:,}",
            'Viewers': f"{n_viewers:,}",
            'Clicks': f"{n_clickers:,}",
            'Click Rate': f"{kpi_row['ctr']:.2f}%",
            'Purchases': f"{n_purchasers:,}",
            'Purchase Rate': f"{kpi_row['retention_pct']:.2f}%",
            'Conversion Rate': f"{kpi_row['conversion_pct']:.2f}%",
            'Revenue/Session': f"${kpi_row['revenue_per_session']:.2f}",
            'Revenue/Purchaser': f"${revenue_per_purchaser:.2f}",
            'Total Revenue': f"${total_revenue:,.0f}"
        })
    
    # Create expandable detailed breakdown
    with st.expander("📋 Detailed Metrics Breakdown", expanded=True):
        # Display side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current System")
            baseline_metrics = detailed_metrics[0]
            st.markdown(f"""
            **Sessions**
            - Total Sessions: **{baseline_metrics['Total Sessions']}**
            
            **Bounce Analysis**
            - Bounce Rate: **{baseline_metrics['Bounce Rate']}**
            - Bounced: **{baseline_metrics['Bounced']}** (left without viewing)
            - Viewers: **{baseline_metrics['Viewers']}** (viewed recommendations)
            
            **Engagement**
            - Clicks: **{baseline_metrics['Clicks']}**
            - Click Rate: **{baseline_metrics['Click Rate']}** (of all sessions)
            
            **Conversion**
            - Purchases: **{baseline_metrics['Purchases']}**
            - Purchase Rate: **{baseline_metrics['Purchase Rate']}** (of all sessions)
            - Conversion Rate: **{baseline_metrics['Conversion Rate']}** (of clickers)
            
            **Revenue**
            - Revenue/Session: **{baseline_metrics['Revenue/Session']}** (per customer, including non-buyers)
            - Revenue/Purchaser: **{baseline_metrics['Revenue/Purchaser']}** (average order value)
            - Total Revenue: **{baseline_metrics['Total Revenue']}** (if all sessions converted)
            """)
        
        with col2:
            st.markdown("#### KNN System")
            knn_metrics = detailed_metrics[1]
            st.markdown(f"""
            **Sessions**
            - Total Sessions: **{knn_metrics['Total Sessions']}**
            
            **Bounce Analysis**
            - Bounce Rate: **{knn_metrics['Bounce Rate']}**
            - Bounced: **{knn_metrics['Bounced']}** (left without viewing)
            - Viewers: **{knn_metrics['Viewers']}** (viewed recommendations)
            
            **Engagement**
            - Clicks: **{knn_metrics['Clicks']}**
            - Click Rate: **{knn_metrics['Click Rate']}** (of all sessions)
            
            **Conversion**
            - Purchases: **{knn_metrics['Purchases']}**
            - Purchase Rate: **{knn_metrics['Purchase Rate']}** (of all sessions)
            - Conversion Rate: **{knn_metrics['Conversion Rate']}** (of clickers)
            
            **Revenue**
            - Revenue/Session: **{knn_metrics['Revenue/Session']}** (per customer, including non-buyers)
            - Revenue/Purchaser: **{knn_metrics['Revenue/Purchaser']}** (average order value)
            - Total Revenue: **{knn_metrics['Total Revenue']}** (if all sessions converted)
            """)
        
        st.markdown("---")
        st.markdown("#### 🎯 Key Formulas")
        st.markdown(f"""
        The metrics are calculated as:
        
        - **Bounce Rate** = (Bounced / Total Sessions) × 100
        - **Click Rate** = (Clicks / Total Sessions) × 100
        - **Purchase Rate** = (Purchases / Total Sessions) × 100
        - **Conversion Rate** = (Purchases / Clicks) × 100
        
        **Three Revenue Metrics:**
        1. **Revenue/Session** = Total Revenue / Total Sessions
           - Includes all sessions (including bounces and non-purchases)
           - **Formula:** (Number of Purchases × Average Price) / Total Sessions
           - **Why it matters?** Shows total revenue impact per customer
        
        2. **Revenue/Purchaser** = Total Revenue / Number of Purchases
           - Only includes actual buyers
           - **Current:** ${avg_purchase_price} (Same for both systems)
           - **Note:** Controlled by "Average purchase price" slider
        
        3. **Total Revenue** = Sum of all purchase prices
           - Total money made from all sessions
           - **This grows** when KNN gets more customers to purchase
        
        **How Price Slider Affects Revenue:**
        - Increasing average price → Higher revenue/session for both systems proportionally
        - Since both systems use same price, revenue growth = Purchase volume growth
        
        **Example with 5,000 sessions @ 23% purchase rate (${avg_purchase_price}/product):**
        - 1,175 purchases at ${avg_purchase_price} each = ${1175 * avg_purchase_price:,} total revenue
        - Revenue/Session = ${1175 * avg_purchase_price:,} / 5,000 = **${1175 * avg_purchase_price / 5000:.2f}/session**
        - Revenue/Purchaser = **${avg_purchase_price:.2f}** (average price)
        
        **With KNN (49% purchase rate, ${avg_purchase_price}/product):**
        - 2,450 purchases at ${avg_purchase_price} each = ${2450 * avg_purchase_price:,} total revenue  
        - Revenue/Session = ${2450 * avg_purchase_price:,} / 5,000 = **${2450 * avg_purchase_price / 5000:.2f}/session**
        - Revenue/Purchaser = **${avg_purchase_price:.2f}** (average price)
        """)
    
    # Charts - 2x2 Grid with Enhanced Visualizations
    st.markdown("### 📈 Comparative Analysis")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data for charts
    baseline_row = kpis_df[kpis_df['policy'] == 'baseline'].iloc[0]
    engine_row = kpis_df[kpis_df['policy'] == 'engine'].iloc[0]
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Color scheme
    colors = ['#ff6b6b', '#51cf66']  # Red for Baseline, Green for KNN
    
    # ===== Chart 1: Purchase Rate (Retention %) =====
    ax = axes[0, 0]
    retention_data = [baseline_row['retention_pct'], engine_row['retention_pct']]
    bars1 = ax.bar(['Current System', 'KNN System'], retention_data, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Purchase Conversion %', fontsize=11, fontweight='bold')
    ax.set_title('💳 Purchase Rate (Customers Who Bought)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(retention_data) * 1.3])
    
    # Add value labels and lift
    lift_retention = ((engine_row['retention_pct'] - baseline_row['retention_pct']) / baseline_row['retention_pct'] * 100) if baseline_row['retention_pct'] > 0 else 0
    for i, (bar, val) in enumerate(zip(bars1, retention_data)):
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}%', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}%\n(+{lift_retention:.2f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ===== Chart 2: Average Revenue per Customer =====
    ax = axes[0, 1]
    revenue_data = [baseline_row['revenue_per_session'], engine_row['revenue_per_session']]
    bars2 = ax.bar(['Current System', 'KNN System'], revenue_data, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Revenue per Session ($)', fontsize=11, fontweight='bold')
    ax.set_title('💰 Revenue Impact ($ per Customer)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(revenue_data) * 1.3])
    
    # Add value labels and lift
    lift_revenue = ((engine_row['revenue_per_session'] - baseline_row['revenue_per_session']) / baseline_row['revenue_per_session'] * 100) if baseline_row['revenue_per_session'] > 0 else 0
    for i, (bar, val) in enumerate(zip(bars2, revenue_data)):
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'${val:.2f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'${val:.2f}\n(+${engine_row["revenue_per_session"] - baseline_row["revenue_per_session"]:.2f})', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ===== Chart 3: OOS Bounce Rate (Lower is Better) =====
    ax = axes[1, 0]
    bounce_data = [baseline_row['oos_bounce_pct'], engine_row['oos_bounce_pct']]
    bars3 = ax.bar(['Current System', 'KNN System'], bounce_data, color=['#ff9999', '#99ff99'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Bounce Rate %', fontsize=11, fontweight='bold')
    ax.set_title('👋 Customer Abandonment (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and reduction
    bounce_reduction = ((baseline_row['oos_bounce_pct'] - engine_row['oos_bounce_pct']) / baseline_row['oos_bounce_pct'] * 100) if baseline_row['oos_bounce_pct'] > 0 else 0
    for i, (bar, val) in enumerate(zip(bars3, bounce_data)):
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            reduction_pp = baseline_row['oos_bounce_pct'] - engine_row['oos_bounce_pct']
            ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%\n(-{reduction_pp:.1f}pp)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ===== Chart 4: Customer Journey Funnel =====
    ax = axes[1, 1]
    categories = ['Viewed\nRecs', 'Clicked', 'Purchased']
    
    # Calculate funnel stages
    baseline_viewed = (1 - baseline_row['oos_bounce_pct']/100) * 100
    baseline_clicked = baseline_row['ctr']
    baseline_purchased = baseline_row['retention_pct']
    
    engine_viewed = (1 - engine_row['oos_bounce_pct']/100) * 100
    engine_clicked = engine_row['ctr']
    engine_purchased = engine_row['retention_pct']
    
    baseline_funnel = [baseline_viewed, baseline_clicked, baseline_purchased]
    engine_funnel = [engine_viewed, engine_clicked, engine_purchased]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars_baseline = ax.bar(x - width/2, baseline_funnel, width, label='Current System', 
                          color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars_engine = ax.bar(x + width/2, engine_funnel, width, label='KNN System', 
                        color='#51cf66', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('% of Customers', fontsize=11, fontweight='bold')
    ax.set_title('📈 Customer Journey (Out-of-Stock Product)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars_baseline, bars_engine]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # ===== FUNNEL VISUALIZATIONS =====
    import plotly.graph_objects as go
    
    st.markdown("---")
    st.markdown("### 🔻 Customer Journey Funnels")
    st.markdown("*From website visit to purchase - Complete flow*")
    
    # The KPI metrics are already aggregated per policy
    # We need to get the actual session counts from the logs for each system
    logs_df = st.session_state.sim_logs
    
    baseline_logs = logs_df[logs_df['policy'] == 'baseline']
    engine_logs = logs_df[logs_df['policy'] == 'engine']
    
    baseline_n_sessions = len(baseline_logs)
    engine_n_sessions = len(engine_logs)
    
    # BASELINE FUNNEL
    # Stage 1: All baseline sessions
    baseline_stage1 = baseline_n_sessions
    # Stage 2: Viewers (didn't bounce)
    baseline_viewers = (baseline_logs['bounced'] == False).sum()
    # Stage 3: Clickers (clicked recommendations)
    baseline_clickers = (baseline_logs['clicked_rec'] == True).sum()
    # Stage 4: Purchasers
    baseline_purchasers = (baseline_logs['purchased'] == True).sum()
    
    # KNN FUNNEL
    # Stage 1: All engine sessions
    knn_stage1 = engine_n_sessions
    # Stage 2: Viewers (didn't bounce)
    knn_viewers = (engine_logs['bounced'] == False).sum()
    # Stage 3: Clickers (clicked recommendations)
    knn_clickers = (engine_logs['clicked_rec'] == True).sum()
    # Stage 4: Purchasers
    knn_purchasers = (engine_logs['purchased'] == True).sum()
    
    # Create two funnel charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Baseline Funnel
        fig_baseline = go.Figure(go.Funnel(
            y=['Session Starts', 'View Recommendations', 'Click Product', 'Purchase'],
            x=[baseline_stage1, baseline_viewers, baseline_clickers, baseline_purchasers],
            textposition='auto',
            marker=dict(color=['#4A90E2', '#7B68EE', '#FF8C42', '#27AE60']),
            text=[
                f"{baseline_stage1:,}<br>({100:.1f}%)",
                f"{baseline_viewers:,}<br>({(baseline_viewers/baseline_stage1)*100:.1f}%)",
                f"{baseline_clickers:,}<br>({(baseline_clickers/baseline_stage1)*100:.1f}%)",
                f"{baseline_purchasers:,}<br>({(baseline_purchasers/baseline_stage1)*100:.1f}%)"
            ],
            hovertemplate='<b>%{y}</b><br>Customers: %{x:,}<extra></extra>',
        ))
        fig_baseline.update_layout(
            title={
                'text': '<b>Current System - Customer Journey</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 14}
            },
            font=dict(size=12),
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_baseline, use_container_width=True, config={'displayModeBar': False})
#-------------    
    with col2:
        # KNN Funnel
        fig_knn = go.Figure(go.Funnel(
            y=['Session Starts', 'View Recommendations', 'Click Product', 'Purchase'],
            x=[knn_stage1, knn_viewers, knn_clickers, knn_purchasers],
            textposition='auto',
            marker=dict(color=['#4A90E2', '#7B68EE', '#FF8C42', '#27AE60']),
            text=[
                f"{knn_stage1:,}<br>({100:.1f}%)",
                f"{knn_viewers:,}<br>({(knn_viewers/knn_stage1)*100:.1f}%)",
                f"{knn_clickers:,}<br>({(knn_clickers/knn_stage1)*100:.1f}%)",
                f"{knn_purchasers:,}<br>({(knn_purchasers/knn_stage1)*100:.1f}%)"
            ],
            hovertemplate='<b>%{y}</b><br>Customers: %{x:,}<extra></extra>',
        ))
        fig_knn.update_layout(
            title={
                'text': '<b>KNN System - Customer Journey</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 14}
            },
            font=dict(size=12),
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_knn, use_container_width=True, config={'displayModeBar': False})

    # Add funnel interpretation
    with st.expander("📊 Understanding the Funnel", expanded=False):
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("**Current System**")
            st.markdown(f"""
            - 👥 Sessions: **{baseline_stage1:,}**
            - 👀 View Recs: **{baseline_viewers:,}** ({(baseline_viewers/baseline_stage1)*100:.1f}%)
            - 🖱️ Click: **{baseline_clickers:,}** ({(baseline_clickers/baseline_stage1)*100:.1f}%)
            - 🛒 Purchase: **{baseline_purchasers:,}** ({(baseline_purchasers/baseline_stage1)*100:.1f}%)
            """)
        
        with col_exp2:
            st.markdown("**KNN System**")
            st.markdown(f"""
            - 👥 Sessions: **{knn_stage1:,}**
            - 👀 View Recs: **{knn_viewers:,}** ({(knn_viewers/knn_stage1)*100:.1f}%)
            - 🖱️ Click: **{knn_clickers:,}** ({(knn_clickers/knn_stage1)*100:.1f}%)
            - 🛒 Purchase: **{knn_purchasers:,}** ({(knn_purchasers/knn_stage1)*100:.1f}%)
            """)
        
        st.markdown("---")
        st.markdown("**Key Improvements**")
        col_imp1, col_imp2, col_imp3, col_imp4 = st.columns(4)
        with col_imp1:
            viewers_diff = knn_viewers - baseline_viewers
            viewers_pct_diff = ((knn_viewers - baseline_viewers)/baseline_viewers)*100 if baseline_viewers > 0 else 0
            st.metric("More Viewers", f"+{viewers_diff:,}", f"+{viewers_pct_diff:.1f}%")
        with col_imp2:
            clicks_diff = knn_clickers - baseline_clickers
            clicks_pct_diff = ((knn_clickers - baseline_clickers)/baseline_clickers)*100 if baseline_clickers > 0 else 0
            st.metric("More Clicks", f"+{clicks_diff:,}", f"+{clicks_pct_diff:.1f}%")
        with col_imp3:
            purchases_diff = knn_purchasers - baseline_purchasers
            purchases_pct_diff = ((knn_purchasers - baseline_purchasers)/baseline_purchasers)*100 if baseline_purchasers > 0 else 0
            st.metric("More Purchases", f"+{purchases_diff:,}", f"+{purchases_pct_diff:.1f}%")
        with col_imp4:
            baseline_rate = baseline_purchasers / baseline_stage1 if baseline_stage1 > 0 else 0
            knn_rate = knn_purchasers / knn_stage1 if knn_stage1 > 0 else 0
            uplift_pct = ((knn_rate / baseline_rate) - 1) * 100 if baseline_rate > 0 else 0
            uplift_mult = (knn_purchasers - baseline_purchasers) / baseline_purchasers if baseline_purchasers > 0 else 0
            st.metric("Purchase Uplift", f"+{uplift_pct:.1f}%", f"+{uplift_mult:.2f}x")
    

    # Logs explorer
    st.markdown("---")
    st.markdown("### 🔍 Session Logs Explorer")
    
    with st.expander("View Detailed Logs"):
        logs_df = st.session_state.sim_logs
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            policy_filter = st.multiselect(
                "Filter by Policy",
                options=logs_df['policy'].unique(),
                default=list(logs_df['policy'].unique())
            )
        
        with col_filter2:
            if not is_ab_test and 'event' in logs_df.columns:
                event_filter = st.multiselect(
                    "Filter by Event",
                    options=logs_df['event'].unique(),
                    default=list(logs_df['event'].unique())
                )
            else:
                event_filter = None
        
        # Apply filters
        filtered_logs = logs_df[logs_df['policy'].isin(policy_filter)]
        if event_filter is not None:
            filtered_logs = filtered_logs[filtered_logs['event'].isin(event_filter)]
        
        st.dataframe(filtered_logs.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(filtered_logs):,} rows")
        
        # Download button
        csv = filtered_logs.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Logs (CSV)",
            data=csv,
            file_name=f"simulation_logs_{seed}.csv",
            mime="text/csv"
        )
    
    # Insights
    st.markdown("---")
    st.markdown("### 💡 Insights")
    
    if is_ab_test:
        # A/B Test Insights
        baseline_row = kpis_df[kpis_df['policy'] == 'baseline']
        engine_row = kpis_df[kpis_df['policy'] == 'engine']
        
        if len(baseline_row) > 0 and len(engine_row) > 0:
            baseline_retention = baseline_row['retention_pct'].values[0]
            engine_retention = engine_row['retention_pct'].values[0]
            baseline_revenue = baseline_row['revenue_per_session'].values[0]
            engine_revenue = engine_row['revenue_per_session'].values[0]
            baseline_bounce = baseline_row['oos_bounce_pct'].values[0]
            engine_bounce = engine_row['oos_bounce_pct'].values[0]
            
            if baseline_retention > 0:
                retention_lift = ((engine_retention - baseline_retention) / baseline_retention) * 100
            else:
                retention_lift = 0
            
            if baseline_revenue > 0:
                revenue_lift = ((engine_revenue - baseline_revenue) / baseline_revenue) * 100
            else:
                revenue_lift = 0
            
            if baseline_bounce > 0:
                bounce_reduction = ((baseline_bounce - engine_bounce) / baseline_bounce) * 100
            else:
                bounce_reduction = 0
            
            col_insight1, col_insight2, col_insight3 = st.columns(3)
            
            with col_insight1:
                st.metric(
                    "Retention Lift",
                    f"{retention_lift:+.1f}%",
                    delta=f"{engine_retention - baseline_retention:.2f} pp",
                    delta_color="off"
                )
            
            with col_insight2:
                st.metric(
                    "Revenue Lift",
                    f"{revenue_lift:+.1f}%",
                    delta=f"${engine_revenue - baseline_revenue:.2f}/session",
                    delta_color="off"
                )
            
            with col_insight3:
                st.metric(
                    "OOS Bounce Reduction",
                    f"{bounce_reduction:+.1f}%",
                    delta=f"{engine_bounce - baseline_bounce:.2f} pp",
                    delta_color="off"
                )
    
    else:
        # Multi-policy Insights
        baseline_retention = kpis_df[kpis_df['policy'] == 'baseline']['retention_pct'].values[0]
        engine_retention = kpis_df[kpis_df['policy'] == 'engine']['retention_pct'].values[0]
        
        if baseline_retention > 0:
            retention_lift = ((engine_retention - baseline_retention) / baseline_retention) * 100
        else:
            retention_lift = 0
        
        baseline_revenue = kpis_df[kpis_df['policy'] == 'baseline']['revenue_per_session'].values[0]
        engine_revenue = kpis_df[kpis_df['policy'] == 'engine']['revenue_per_session'].values[0]
        
        if baseline_revenue > 0:
            revenue_lift = ((engine_revenue - baseline_revenue) / baseline_revenue) * 100
        else:
            revenue_lift = 0
        
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            st.metric(
                "Retention Lift vs Baseline",
                f"{retention_lift:+.1f}%",
                delta=f"{engine_retention - baseline_retention:.2f} pp"
            )
        
        with col_insight2:
            st.metric(
                "Revenue Lift vs Baseline",
                f"{revenue_lift:+.1f}%",
                delta=f"${engine_revenue - baseline_revenue:.2f}/session"
            )
        
        with col_insight3:
            engine_ctr = kpis_df[kpis_df['policy'] == 'engine']['ctr'].values[0]
            st.metric(
                "Engine CTR",
                f"{engine_ctr:.2f}%"
            )

else:
    st.info("👆 Configure parameters in the sidebar and click 'Start Simulation' to begin.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Synthetic Users Simulator • Built for Fashion Recommender System</p>
    <p>A/B test recommendation policies with configurable user behavior models</p>
</div>
""", unsafe_allow_html=True)

# FAQ Section
st.markdown("---")

# Show appropriate FAQ based on mode
if use_dynamic_bounce:
    st.markdown("## ❓ FAQ - Quality-Based Mode")
    
    with st.expander("❓ What is k-Nearest Neighbours (KNN)?", expanded=False):
        st.markdown("""
        KNN finds products **visually similar to what customers were looking for**.
        
        When a product is out-of-stock, KNN recommends alternatives that look and feel similar—increasing the chance customers stay and purchase.
        """)
    
    with st.expander("🎯 How do bounce rates work?", expanded=False):
        st.markdown("""
        **Formula (Custom, not academic):**
        ```
        Bounce = Base - (Quality - 0.5) × Sensitivity
        ```
        
        **Why custom?** Research shows quality affects bounce (Fitzsimons 2000, Hoang & Breugelmans 2023) but provides no precise formula.
        
        **Examples (Base=65%, Sensitivity=0.5):**
        - Quality 0.2 (poor): 72.5% bounce
        - Quality 0.5 (neutral): 65% bounce
        - Quality 0.8 (good): 57.5% bounce
        """)
    
    with st.expander("🏛️ How does the Baseline (Current) System work?", expanded=False):
        st.markdown("""
        **Baseline uses fixed academic values** (random recommendations):
        - Bounce: 65% (Fitzsimons 2000)
        - Click: ~25% avg (Cremonesi 2010)
        - Purchase: ~10% avg (Industry standard)
        
        **KNN-only parameters** (adjustable):
        - User Sensitivity, KNN Base Bounce, KNN Accuracy
        - Click Uplift, Purchase Uplift
        
        **Applies to both:**
        - Price Tolerance, Average Purchase Price
        """)
    
    with st.expander("💡 How does the simulator work?", expanded=False):
        st.markdown("""
        1. Load 52,527 front-facing product images from catalog (analyzed from 52,712 total embeddings)
        2. For each simulated customer at OOS:
           - Generate 5 recommendations (random vs KNN)
           - Measure recommendation quality (similarity, price match, availability)
           - Calculate bounce rate based on quality
           - Simulate clicks and purchases
        3. Aggregate results across all sessions
        4. Compare Current System vs KNN System
        """)
    
    with st.expander("📊 What do the metrics mean?", expanded=False):
        st.markdown("""
        | Metric | Meaning |
        |--------|---------|
        | **Bounce Rate** | % of customers who leave without viewing recommendations |
        | **Click Rate** | % of customers who click on a recommendation |
        | **Purchase Rate** | % of customers who actually buy |
        | **Revenue/Session** | Average dollars earned per customer |
        """)
    
    with st.expander("🔧 Parameter Guide", expanded=False):
        st.markdown("""
        **User Sensitivity (KNN):** How much quality affects bounce (default 0.5)
        
        **KNN Base Bounce:** Starting bounce % for KNN before quality adjustment (default 50%)
        
        **KNN Accuracy:** Similarity threshold to count as "good" rec (default 0.6)
        
        **Price Tolerance:** Acceptable price range ±% (default ±20%)
        
        **Click/Purchase Uplift:** Multiplier vs baseline rates (KNN only)
        """)
    
    with st.expander("📊 What is a Quality Score?", expanded=False):
        st.markdown("""
        Quality Score (0.0–1.0) measures how good the 5 recommendations are.
        
        **Each recommendation passes if it meets ALL 3 criteria:**
        1. Similarity ≥ KNN Accuracy threshold
        2. Price within ± Price Tolerance
        3. Item is in stock
        
        **Quality Score = (# passing all 3) / 5**
        
        Examples: 5/5 pass = 1.0 (perfect), 3/5 pass = 0.6 (good), 0/5 pass = 0.0 (terrible)
        """)


    with st.expander("🔢 What does the 0.5 in the formula mean?", expanded=False):
        st.markdown("""
        Formula: `Bounce = Base - (Quality - 0.5) × Sensitivity`
        
        **0.5 = Neutral point** where quality has no effect on bounce.
        
        - Quality > 0.5: Bounce decreases (better than baseline)
        - Quality = 0.5: Bounce stays at baseline
        - Quality < 0.5: Bounce increases (worse than baseline)
        
        **Example (Base=65%, Sensitivity=0.5):**
        - Quality 0.2 → 80% bounce (poor recs hurt)
        - Quality 0.5 → 65% bounce (neutral)
        - Quality 0.8 → 50% bounce (good recs help)
        """)
    
    with st.expander("📚 Why was this formula created?", expanded=False):
        st.markdown("""
        **Custom formula, not academic.**
        
        Research shows quality affects bounce (Fitzsimons 2000, Hoang & Breugelmans 2023) but no precise formula exists.
        
        **Our approach:**
        - Respects academic findings (65% baseline, quality reduces to ~40%)
        - Linear for simplicity
        - Centered at 0.5 for symmetry
        - Sensitivity allows calibration from real A/B tests
        """)

else:
    st.markdown("## ❓ FAQ - Legacy Mode (Fixed Bounce Rates)")
    
    with st.expander("❓ What is Legacy Mode?", expanded=False):
        st.markdown("""
        Legacy Mode lets you set bounce rates with sliders for **scenario testing and what-if analysis**.
        
        You manually control:
        - How many customers bounce with your current system
        - How many bounce with KNN
        
        This is useful for exploring different scenarios, but not recommended for realistic predictions.
        """)
    
    with st.expander("🎯 How do bounce rates work in this mode?", expanded=False):
        st.markdown("""
        **You set bounce rates directly with sliders.**
        
        **Bounce Calculation:**
        - You control exactly what % of customers bounce for each system
        - No formula — the values you set are used directly
        - Useful for scenario testing or matching historical data
        
        Set the bounce rate you expect for:
        - **Current System**: Your baseline (random recommendations, typically 60-75%)
        - **KNN System**: Performance with better recommendations (typically 35-50%)
        
        The simulator uses these fixed values for all sessions.
        """)
    
    with st.expander("💡 When should I use Legacy Mode?", expanded=False):
        st.markdown("""
        Use Legacy Mode to:
        - Test specific "what-if" scenarios
        - Compare to past assumptions
        - Explore extreme cases (very optimistic/pessimistic)
        - Scenario planning before running real tests
        
        For realistic predictions, use **Quality-Based Mode** instead.
        """)
    
    with st.expander("📊 What do the metrics mean?", expanded=False):
        st.markdown("""
        | Metric | Meaning |
        |--------|---------|
        | **Bounce Rate** | % you set with the sliders (fixed for all sessions) |
        | **Click Rate** | % of customers who click on a recommendation |
        | **Purchase Rate** | % of customers who actually buy |
        | **Revenue/Session** | Average dollars earned per customer |
        """)
    
    with st.expander("🔧 How do I use each parameter?", expanded=False):
        st.markdown("""
        **Current System Bounce %**
        - Typical: 60-75% for random recommendations
        - This is your baseline assumption
        
        **KNN System Bounce %**
        - Typical: 35-50% with good recommendations
        - This is your projected improvement
        
        **Other sliders** (Click Uplift, Purchase Uplift, Price)
        - Work the same in both modes
        """)

with st.expander("🎯 What exactly are we A/B testing?", expanded=False):
    st.markdown("""
    **The Scenario:**
    A customer finds a product they love but it's out of stock. We need to show them alternatives.
    
    **The Question:**
    Which system recovers more lost sales?
    - **Current System**: Random products (poor quality)
    - **KNN System**: Semantically similar products (high quality)
    
    **The Value:**
    Out-of-stock is critical. Good recommendations can recover lost sales and customer lifetime value.
    """)


# Render Sources & Assumptions section at the bottom
render_sources_and_assumptions()
