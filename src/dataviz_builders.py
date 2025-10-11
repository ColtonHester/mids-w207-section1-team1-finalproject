import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings('ignore')


def plot_geographic_maps(df: pd.DataFrame, lat_col='LATITUDE', lon_col='LONGITUDE',
                         target: str = None, figsize: tuple = (18, 8)):
    """
    Create side-by-side geographic maps showing fire density and distribution by target class.

    Parameters:
    -----------
    df : Training data
    lat_col : Latitude column
    lon_col : Longitude column
    target : Target variable for classification
    figsize : tuple
        Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig, hspace=0.2, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])

    hexbin = ax1.hexbin(df[lon_col], df[lat_col], gridsize=60,
                        cmap='YlOrRd', mincnt=1, alpha=0.9, edgecolors='black', linewidths=0.2)
    ax1.set_xlabel('Longitude (°W)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    ax1.set_title('Wildfire Density Map\n(Hexagonal binning shows fire concentration)',
                  fontsize=14, fontweight='bold', pad=15)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    cbar1 = plt.colorbar(hexbin, cax=cax1, label='Fire Count')
    cbar1.ax.tick_params(labelsize=10)

    # Add approximate state labels for context
    annotations = [
        (-122, 38, 'CA', 'black'),
        (-112, 40, 'UT/NV', 'black'),
        (-105, 39, 'CO', 'black'),
        (-100, 35, 'OK/TX', 'black'),
        (-84, 33, 'GA/AL', 'black'),
        (-82, 28, 'FL', 'black'),
    ]

    for lon, lat, label, color in annotations:
        ax1.annotate(label, xy=(lon, lat), fontsize=10, fontweight='bold',
                     color=color, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.7, edgecolor='black', linewidth=1))

    ax2 = fig.add_subplot(gs[0, 1])

    if target and target in df.columns:
        unique_classes = sorted(df[target].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))

        for idx, cls in enumerate(unique_classes):
            mask = df[target] == cls
            sample_size = min(1000, mask.sum())  # Limit points for clarity
            if sample_size > 0:
                sample_idx = df[mask].sample(n=sample_size, random_state=42).index

                ax2.scatter(df.loc[sample_idx, lon_col], df.loc[sample_idx, lat_col],
                            alpha=0.5, s=20, c=[colors[idx]], label=f'Class {cls}',
                            edgecolors='black', linewidth=0.3, rasterized=True)

        ax2.legend(fontsize=11, markerscale=2, loc='upper left',
                   framealpha=0.95, title=target, title_fontsize=12)
    else:
        # Sample for visualization
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        ax2.scatter(df_sample[lon_col], df_sample[lat_col],
                    alpha=0.3, s=15, c='steelblue', edgecolors='none')

    ax2.set_xlabel('Longitude (°W)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    ax2.set_title('Fire Locations by Target Class\n(Sampled points colored by fire size class)',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regional_distributions(df: pd.DataFrame, lat_col: str = 'LATITUDE', lon_col: str = 'LONGITUDE',
                                target: str = None, figsize=(18, 6)):
    """
    Create side-by-side distribution plots showing fire counts by geographic regions.

    Parameters:
    -----------
    df :Training data
    lat_col : Latitude column name
    lon_col : Longitude column name
    target : Target variable for classification
    figsize : Figure size (width, height)
    """

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.4)

    # Create regional bins
    lat_bins = [18, 30, 37, 42, 49, 72]
    lat_labels = ['South\n(18-30°)', 'Southeast\n(30-37°)', 'Central\n(37-42°)',
                  'North-Central\n(42-49°)', 'North\n(49-72°)']

    lon_bins = [-180, -125, -110, -100, -90, -75, -65]
    lon_labels = ['Pacific\nCoast', 'Mountain\nWest', 'Great\nPlains',
                  'Midwest', 'Southeast', 'Atlantic']

    df['lat_region'] = pd.cut(df[lat_col], bins=lat_bins, labels=lat_labels)
    df['lon_region'] = pd.cut(df[lon_col], bins=lon_bins, labels=lon_labels)

    ax1 = fig.add_subplot(gs[0, 0])

    regional_stats = df['lat_region'].value_counts().sort_index()

    colors_regional = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bars = ax1.bar(range(len(regional_stats)), regional_stats.values,
                   color=colors_regional, edgecolor='black', alpha=0.8, linewidth=1.5)

    ax1.set_xticks(range(len(regional_stats)))
    ax1.set_xticklabels(regional_stats.index, fontsize=10, rotation=0, ha='center')
    ax1.set_ylabel('Fire Count', fontsize=12, fontweight='bold')
    ax1.set_title('Fire Distribution by\nLatitude Region',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])

    lon_stats = df['lon_region'].value_counts().sort_index()

    colors_lon = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars_lon = ax2.bar(range(len(lon_stats)), lon_stats.values,
                       color=colors_lon, edgecolor='black', alpha=0.8, linewidth=1.5)

    ax2.set_xticks(range(len(lon_stats)))
    ax2.set_xticklabels(lon_stats.index, fontsize=10, rotation=45, ha='right')
    ax2.set_ylabel('Fire Count', fontsize=12, fontweight='bold')
    ax2.set_title('Fire Distribution by\nLongitude Region',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars_lon:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])

    if target and target in df.columns:
        # MTBS region-specific thresholds based on longitude (VECTORIZED)
        # Western US (lon < -100°W): ≥1000 acres (Class F+)
        # Eastern US (lon ≥ -100°W): ≥500 acres (Class E+)

        # Vectorized approach - much faster!
        western_mask = df[lon_col] < -100
        eastern_mask = df[lon_col] >= -100

        # Initialize all as below threshold
        df['mtbs_category'] = 'Below MTBS\nThreshold'

        # Western US: Class F+ is above threshold
        df.loc[western_mask & (df[target] >= 'F'), 'mtbs_category'] = 'Above MTBS\nThreshold'

        # Eastern US: Class E+ is above threshold
        df.loc[eastern_mask & (df[target] >= 'E'), 'mtbs_category'] = 'Above MTBS\nThreshold'

        # Calculate regional composition within each MTBS category
        mtbs_region = pd.crosstab(df['mtbs_category'], df['lat_region'], normalize='index') * 100

        # Reorder to have Below first, Above second
        mtbs_order = ['Below MTBS\nThreshold', 'Above MTBS\nThreshold']
        mtbs_region = mtbs_region.reindex(mtbs_order)

        # Define region colors (consistent with first plot)
        region_colors = {
            'South\n(18-30°)': '#d62728',
            'Southeast\n(30-37°)': '#ff7f0e',
            'Central\n(37-42°)': '#2ca02c',
            'North-Central\n(42-49°)': '#1f77b4',
            'North\n(49-72°)': '#9467bd'
        }

        # Create stacked bar chart
        bottom = np.zeros(len(mtbs_region.index))
        bar_width = 0.6

        for region in mtbs_region.columns:
            values = mtbs_region[region].values
            bars = ax3.bar(range(len(mtbs_region.index)), values, bar_width,
                           bottom=bottom, label=region,
                           color=region_colors.get(region, '#808080'),
                           edgecolor='black', linewidth=1.5)

            # Add percentage labels on segments (only if segment is large enough)
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 5:  # Only show label if segment is >5%
                    label_y = bottom[i] + val / 2
                    ax3.text(bar.get_x() + bar.get_width() / 2., label_y,
                             f'{val:.1f}%',
                             ha='center', va='center', fontsize=9,
                             fontweight='bold', color='white')

            bottom += values

        ax3.set_ylabel('Regional Composition (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Regional Distribution:\nMTBS Threshold Analysis\n(West: ≥1000ac, East: ≥500ac)',
                      fontsize=12, fontweight='bold', pad=15)
        ax3.set_xticks(range(len(mtbs_region.index)))
        ax3.set_xticklabels(mtbs_region.index, fontsize=10, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9,
                   framealpha=0.95, title='Region', title_fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

    else:
        # Show message if no target
        ax3.text(0.5, 0.5, 'No target\nvariable\nprovided',
                 ha='center', va='center', fontsize=12,
                 transform=ax3.transAxes, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('MTBS Threshold\nAnalysis', fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()


def create_geo_insights_table(df, lat_col='LATITUDE', lon_col='LONGITUDE',
                              target=None):
    """
    Create summary table
    """
    # Define regions
    lat_bins = [18, 30, 37, 42, 49, 72]
    lat_labels = ['South (18-30°)', 'Southeast (30-37°)', 'Central (37-42°)',
                  'North-Central (42-49°)', 'North (49-72°)']

    df['lat_region'] = pd.cut(df[lat_col], bins=lat_bins, labels=lat_labels)

    # statistics by region
    summary_stats = []

    for region in lat_labels:
        region_data = df[df['lat_region'] == region]

        stats = {
            'Region': region,
            'Fire Count': len(region_data),
            'Percentage': len(region_data) / len(df) * 100,
            'Avg Latitude': region_data[lat_col].mean(),
            'Avg Longitude': region_data[lon_col].mean(),
        }

        if target and target in df.columns:
            # Most common target class
            mode_class = region_data[target].mode()[0] if len(region_data) > 0 else None
            stats['Most Common Class'] = mode_class
            stats['Class %'] = (region_data[target] == mode_class).sum() / len(region_data) * 100 if len(
                region_data) > 0 else 0

        summary_stats.append(stats)

    return pd.DataFrame(summary_stats)


def plot_fire_cause_analysis(df, cause_col='NWCG_CAUSE_CLASSIFICATION',
                             target='FIRE_SIZE_CLASS', lat_col='LATITUDE',
                             lon_col='LONGITUDE', figsize=(18, 5)):
    """
    Comprehensive analysis of fire causes and their relationship to fire severity.

    Parameters:
    -----------
    df : DataFrame
        Training data
    cause_col : str
        Fire cause classification column
    target : str
        Fire size class target variable
    lat_col, lon_col : str
        Location columns
    """
    # Create regional bins for analysis
    lat_bins = [18, 30, 37, 42, 49, 72]
    lat_labels = ['South', 'Southeast', 'Central', 'North-Central', 'North']
    df['lat_region'] = pd.cut(df[lat_col], bins=lat_bins, labels=lat_labels)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])

    if cause_col in df.columns:
        causes = df[cause_col].unique()
        colors_map = dict(zip(causes, plt.cm.Set2(np.linspace(0, 1, len(causes)))))

        # Sample for visualization
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        for cause in causes:
            mask = df_sample[cause_col] == cause
            ax1.scatter(df_sample.loc[mask, lon_col], df_sample.loc[mask, lat_col],
                        alpha=0.5, s=20, c=[colors_map[cause]], label=cause,
                        edgecolors='black', linewidth=0.3)

        ax1.set_xlabel('Longitude (°W)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
        ax1.set_title('Geographic Distribution of Fire Causes\n(Are certain regions dominated by specific causes?)',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=10, markerscale=2)
        ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])

    if cause_col in df.columns and target in df.columns:
        # Create numeric mapping for size analysis
        class_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df['size_numeric'] = df[target].map(class_mapping)

        # Define high-impact fires (Class D+)
        df['high_impact'] = df[target] >= 'D'

        # Count by cause for high-impact vs all fires
        all_fires = df[cause_col].value_counts()
        high_impact_fires = df[df['high_impact']][cause_col].value_counts()

        # Calculate percentage
        high_impact_pct = (high_impact_fires / all_fires * 100).fillna(0)

        # Plot
        bars = ax2.bar(range(len(high_impact_pct)), high_impact_pct.values,
                       color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)

        ax2.set_xticks(range(len(high_impact_pct)))
        ax2.set_xticklabels(high_impact_pct.index, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('% High-Impact (≥Class D)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Fire Cause', fontsize=11, fontweight='bold')
        ax2.set_title('High-Impact Fire Rate by Cause\n(Which causes lead to large fires?)',
                      fontsize=12, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_infrastructure_effectiveness(df, firestation_col:str='No_FireStation_20.0km',
                                      preparedness_col:str='GACC_PL', target:str='FIRE_SIZE_CLASS',
                                      figsize=(18, 5)):
    """
    Analyze relationship between infrastructure/preparedness and fire outcomes.

    Parameters:
    -----------
    df : Training data
    firestation_col : Number of fire stations within 20km
    preparedness_col : GACC preparedness level (1-5)
    target : Fire size class target variable
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.35)

    # Create numeric fire size for correlation analysis
    class_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['size_numeric'] = df[target].map(class_mapping)

    # binned fire stations for analysis
    df['fs_binned'] = pd.cut(df[firestation_col], bins=[-1, 0, 2, 5, 10, 100],
                             labels=['0', '1-2', '3-5', '6-10', '10+'])

    # Define high-impact fires
    df['high_impact'] = df[target] >= 'D'

    ax1 = fig.add_subplot(gs[0, 0])

    if firestation_col in df.columns and target in df.columns:
        high_impact_rate = df.groupby('fs_binned')['high_impact'].mean() * 100

        bars = ax1.bar(range(len(high_impact_rate)), high_impact_rate.values,
                       color='crimson', edgecolor='black', linewidth=1.5, alpha=0.8)

        ax1.set_xticks(range(len(high_impact_rate)))
        ax1.set_xticklabels(high_impact_rate.index, fontsize=10)
        ax1.set_ylabel('% High-Impact Fires (≥D)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Fire Stations within 20km', fontsize=11, fontweight='bold')
        ax1.set_title('High-Impact Fire Rate vs Infrastructure\n(Does proximity to fire stations help?)',
                      fontsize=12, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])

    if preparedness_col in df.columns and target in df.columns:
        # Filter out NaN preparedness values
        df_prep_clean = df[df[preparedness_col].notna()]
        high_impact_prep = df_prep_clean.groupby(preparedness_col)['high_impact'].mean() * 100

        bars2 = ax2.bar(range(len(high_impact_prep)), high_impact_prep.values,
                        color='darkorange', edgecolor='black', linewidth=1.5, alpha=0.8)

        ax2.set_xticks(range(len(high_impact_prep)))
        ax2.set_xticklabels([str(int(level)) for level in high_impact_prep.index], fontsize=10)
        ax2.set_ylabel('% High-Impact Fires (≥D)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('GACC Preparedness Level', fontsize=11, fontweight='bold')
        ax2.set_title('High-Impact Fire Rate vs Preparedness\n(Does readiness matter?)',
                      fontsize=12, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])

    if firestation_col in df.columns and preparedness_col in df.columns:
        # Calculate correlations (removing NaN values)
        df_corr = df[[firestation_col, preparedness_col, 'size_numeric']].dropna()
        corr_fs = df_corr[[firestation_col, 'size_numeric']].corr().iloc[0, 1]
        corr_prep = df_corr[[preparedness_col, 'size_numeric']].corr().iloc[0, 1]

        # Create bar chart of correlations
        correlations = pd.Series({
            'Fire Stations\nvs Size': corr_fs,
            'Preparedness\nvs Size': corr_prep
        })

        colors_corr = ['green' if x < 0 else 'red' for x in correlations.values]
        bars3 = ax3.bar(range(len(correlations)), correlations.values,
                        color=colors_corr, edgecolor='black', linewidth=1.5, alpha=0.7)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xticks(range(len(correlations)))
        ax3.set_xticklabels(correlations.index, fontsize=10)
        ax3.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax3.set_title('Correlation with Fire Size\n(Negative = Helps)',
                      fontsize=12, fontweight='bold', pad=15)
        ax3.set_ylim(-0.3, 0.3)
        ax3.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()