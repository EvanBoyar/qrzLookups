import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import pytz
from tzlocal import get_localzone
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_process_data(filepath):
    """Load CSV and process datetime columns for both UTC and local time"""
    # Read CSV
    df = pd.read_csv(filepath)

    # Convert Time column to datetime (assuming UTC)
    df['Time_UTC'] = pd.to_datetime(df['Time'], utc=True)

    # Get local timezone and convert
    local_tz = get_localzone()
    df['Time_Local'] = df['Time_UTC'].dt.tz_convert(local_tz)

    # Sort by time to ensure proper ordering
    df = df.sort_values('Time_UTC')

    # Calculate the rate of change (derivative)
    df['Value_Change'] = df['Hits'].diff()
    df['Time_Delta'] = df['Time_UTC'].diff().dt.total_seconds() / 3600  # in hours
    df['Rate_Per_Hour'] = df['Value_Change'] / df['Time_Delta']

    # Extract time components for UTC
    df['Hour_UTC'] = df['Time_UTC'].dt.hour
    df['Day_of_Week_UTC'] = df['Time_UTC'].dt.day_name()
    df['Date_UTC'] = df['Time_UTC'].dt.date

    # Extract time components for Local
    df['Hour_Local'] = df['Time_Local'].dt.hour
    df['Day_of_Week_Local'] = df['Time_Local'].dt.day_name()
    df['Date_Local'] = df['Time_Local'].dt.date

    # Get timezone names for labeling
    utc_name = "UTC"
    local_name = str(local_tz)

    return df, utc_name, local_name

def plot_raw_values(df, utc_name, figsize=(14, 6)):
    """Plot raw values over time in linear scale (UTC only)"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(df['Time_UTC'], df['Hits'], marker='o', linewidth=2, markersize=4, color='steelblue')
    ax.set_xlabel(f'Date/Time ({utc_name})', fontsize=12)
    ax.set_ylabel('Hits', fontsize=12)
    ax.set_title(f'Raw Values Over Time - {utc_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    if len(df) <= 30:
        for idx, row in df.iterrows():
            ax.annotate(f'{row["Hits"]:,}',
                        xy=(row['Time_UTC'], row['Hits']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    plt.tight_layout()
    return fig


def plot_recent_raw_values(df, utc_name, local_name, days=30, figsize=(14, 12)):
    """Plot the last N days of raw values in linear scale for both UTC and local time"""
    cutoff = df['Time_UTC'].max() - pd.Timedelta(days=days)
    df_recent = df[df['Time_UTC'] >= cutoff]

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    for ax, time_col, color, label in [
        (axes[0], 'Time_UTC',   'steelblue',  utc_name),
        (axes[1], 'Time_Local', 'darkgreen',  local_name),
    ]:
        ax.plot(df_recent[time_col], df_recent['Hits'], marker='o', linewidth=2, markersize=4, color=color)
        ax.set_xlabel(f'Date/Time ({label})', fontsize=12)
        ax.set_ylabel('Hits', fontsize=12)
        ax.set_title(f'Raw Values — Last {days} Days ({label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    return fig

def plot_raw_values_log(df, utc_name, figsize=(14, 6)):
    """Plot raw values over time in logarithmic scale (UTC only)"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_min_exp = int(np.floor(np.log10(df['Hits'].min())))
    y_max_exp = int(np.ceil(np.log10(df['Hits'].max())))
    y_min = 10 ** y_min_exp
    y_max = 10 ** y_max_exp
    major_ticks = [10**i for i in range(y_min_exp, y_max_exp + 1)]
    minor_ticks = [m * 10**i for i in range(y_min_exp, y_max_exp) for m in range(2, 10)]

    ax.plot(df['Time_UTC'], df['Hits'], marker='o', linewidth=2, markersize=4, color='steelblue')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f'Date/Time ({utc_name})', fontsize=12)
    ax.set_ylabel('Hits (Log Scale)', fontsize=12)
    ax.set_title(f'Raw Values Over Time (Log Scale) - {utc_name}', fontsize=14, fontweight='bold')
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls="-", color="#888888", alpha=0.6)
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f'$10^{{{int(np.log10(t))}}}$' for t in major_ticks])
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels([f'{t:,}' for t in minor_ticks], minor=True)
    ax.tick_params(axis='y', which='minor', labelsize=11, labelcolor='#222222')

    plt.tight_layout()
    return fig

def plot_hourly_rate_analysis(df, utc_name, local_name, figsize=(14, 6)):
    """Analyze average hourly rate of change for both UTC and local time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Filter out NaN values from rate calculations
    df_rate = df.dropna(subset=['Rate_Per_Hour'])

    # UTC analysis
    if len(df_rate) > 0:
        hourly_stats_utc = df_rate.groupby('Hour_UTC')['Rate_Per_Hour'].agg(['mean', 'std', 'count'])
        hourly_stats_utc = hourly_stats_utc.reset_index()

        bars1 = ax1.bar(hourly_stats_utc['Hour_UTC'], hourly_stats_utc['mean'],
                       yerr=hourly_stats_utc['std'], capsize=5, alpha=0.7)
        ax1.set_xlabel(f'Hour of Day ({utc_name})', fontsize=12)
        ax1.set_ylabel('Average Rate of Change (per hour)', fontsize=12)
        ax1.set_title(f'Average Hourly Rate of Change - {utc_name}', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(0, 24))
        ax1.grid(True, alpha=0.3, axis='y')

        # Color bars by magnitude
        if len(hourly_stats_utc) > 0 and hourly_stats_utc['mean'].max() > 0:
            colors = plt.cm.coolwarm(hourly_stats_utc['mean'] / hourly_stats_utc['mean'].max())
            for bar, color in zip(bars1, colors):
                bar.set_color(color)

    # Local time analysis
    if len(df_rate) > 0:
        hourly_stats_local = df_rate.groupby('Hour_Local')['Rate_Per_Hour'].agg(['mean', 'std', 'count'])
        hourly_stats_local = hourly_stats_local.reset_index()

        bars2 = ax2.bar(hourly_stats_local['Hour_Local'], hourly_stats_local['mean'],
                       yerr=hourly_stats_local['std'], capsize=5, alpha=0.7)
        ax2.set_xlabel(f'Hour of Day ({local_name})', fontsize=12)
        ax2.set_ylabel('Average Rate of Change (per hour)', fontsize=12)
        ax2.set_title(f'Average Hourly Rate of Change - {local_name}', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(0, 24))
        ax2.grid(True, alpha=0.3, axis='y')

        # Color bars by magnitude
        if len(hourly_stats_local) > 0 and hourly_stats_local['mean'].max() > 0:
            colors = plt.cm.coolwarm(hourly_stats_local['mean'] / hourly_stats_local['mean'].max())
            for bar, color in zip(bars2, colors):
                bar.set_color(color)

    plt.tight_layout()
    return fig

def plot_activity_heatmaps(df, utc_name, local_name, figsize=(16, 6)):
    """Create activity heatmaps for both UTC and local time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # UTC heatmap
    if len(df['Date_UTC'].unique()) > 7:  # Need at least a week of data for meaningful heatmap
        df_pivot_utc = df.pivot_table(values='Value_Change',
                                      index=df['Hour_UTC'],
                                      columns=df['Date_UTC'],
                                      aggfunc='sum')

        if not df_pivot_utc.empty:
            sns.heatmap(df_pivot_utc, cmap='YlOrRd', ax=ax1,
                       cbar_kws={'label': 'Value Change'})
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel(f'Hour of Day ({utc_name})', fontsize=12)
            ax1.set_title(f'Daily Activity Heatmap - {utc_name}', fontsize=13, fontweight='bold')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for heatmap\n(need more than 7 days)',
                ha='center', va='center', fontsize=12)
        ax1.set_title(f'Daily Activity Heatmap - {utc_name}', fontsize=13, fontweight='bold')

    # Local time heatmap
    if len(df['Date_Local'].unique()) > 7:
        df_pivot_local = df.pivot_table(values='Value_Change',
                                        index=df['Hour_Local'],
                                        columns=df['Date_Local'],
                                        aggfunc='sum')

        if not df_pivot_local.empty:
            sns.heatmap(df_pivot_local, cmap='YlOrRd', ax=ax2,
                       cbar_kws={'label': 'Value Change'})
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel(f'Hour of Day ({local_name})', fontsize=12)
            ax2.set_title(f'Daily Activity Heatmap - {local_name}', fontsize=13, fontweight='bold')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for heatmap\n(need more than 7 days)',
                ha='center', va='center', fontsize=12)
        ax2.set_title(f'Daily Activity Heatmap - {local_name}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_day_of_week_heatmaps(df, utc_name, local_name, figsize=(16, 8)):
    """Create day-of-week activity heatmaps showing patterns by weekday"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Define day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # UTC day-of-week heatmap
    df_dow_utc = df.groupby(['Day_of_Week_UTC', 'Hour_UTC'])['Rate_Per_Hour'].mean().reset_index()
    if not df_dow_utc.empty:
        df_pivot_dow_utc = df_dow_utc.pivot(index='Day_of_Week_UTC',
                                            columns='Hour_UTC',
                                            values='Rate_Per_Hour')
        # Reindex to ensure day order
        df_pivot_dow_utc = df_pivot_dow_utc.reindex([d for d in day_order if d in df_pivot_dow_utc.index])

        if not df_pivot_dow_utc.empty:
            sns.heatmap(df_pivot_dow_utc, cmap='YlOrRd', ax=ax1,
                       cbar_kws={'label': 'Avg Rate/Hour'},
                       fmt='.1f', annot=len(df_pivot_dow_utc) <= 7)
            ax1.set_xlabel(f'Hour of Day ({utc_name})', fontsize=12)
            ax1.set_ylabel('Day of Week', fontsize=12)
            ax1.set_title(f'Average Activity by Day of Week - {utc_name}', fontsize=13, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        ax1.set_title(f'Average Activity by Day of Week - {utc_name}', fontsize=13, fontweight='bold')

    # Local time day-of-week heatmap
    df_dow_local = df.groupby(['Day_of_Week_Local', 'Hour_Local'])['Rate_Per_Hour'].mean().reset_index()
    if not df_dow_local.empty:
        df_pivot_dow_local = df_dow_local.pivot(index='Day_of_Week_Local',
                                                columns='Hour_Local',
                                                values='Rate_Per_Hour')
        # Reindex to ensure day order
        df_pivot_dow_local = df_pivot_dow_local.reindex([d for d in day_order if d in df_pivot_dow_local.index])

        if not df_pivot_dow_local.empty:
            sns.heatmap(df_pivot_dow_local, cmap='YlOrRd', ax=ax2,
                       cbar_kws={'label': 'Avg Rate/Hour'},
                       fmt='.1f', annot=len(df_pivot_dow_local) <= 7)
            ax2.set_xlabel(f'Hour of Day ({local_name})', fontsize=12)
            ax2.set_ylabel('Day of Week', fontsize=12)
            ax2.set_title(f'Average Activity by Day of Week - {local_name}', fontsize=13, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        ax2.set_title(f'Average Activity by Day of Week - {local_name}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig

def generate_summary_stats(df, utc_name, local_name):
    """Generate summary statistics"""
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal Records: {len(df)}")
    print(f"Date Range ({utc_name}): {df['Time_UTC'].min()} to {df['Time_UTC'].max()}")
    print(f"Date Range ({local_name}): {df['Time_Local'].min()} to {df['Time_Local'].max()}")
    print(f"Total Duration: {df['Time_UTC'].max() - df['Time_UTC'].min()}")

    print(f"\nValue Statistics:")
    print(f"  - Starting Value: {df['Hits'].iloc[0]:,}")
    print(f"  - Ending Value: {df['Hits'].iloc[-1]:,}")
    print(f"  - Total Increase: {df['Hits'].iloc[-1] - df['Hits'].iloc[0]:,}")
    print(f"  - Min Value: {df['Hits'].min():,}")
    print(f"  - Max Value: {df['Hits'].max():,}")
    print(f"  - Mean Value: {df['Hits'].mean():,.2f}")
    print(f"  - Median Value: {df['Hits'].median():,.2f}")

    # Rate statistics (excluding NaN)
    df_rate = df.dropna(subset=['Rate_Per_Hour'])
    if len(df_rate) > 0:
        print(f"\nRate of Change Statistics (per hour):")
        print(f"  - Mean Rate: {df_rate['Rate_Per_Hour'].mean():,.2f}")
        print(f"  - Median Rate: {df_rate['Rate_Per_Hour'].median():,.2f}")
        print(f"  - Max Rate: {df_rate['Rate_Per_Hour'].max():,.2f}")
        print(f"  - Min Rate: {df_rate['Rate_Per_Hour'].min():,.2f}")

        # Best performing hours for both timezones
        print(f"\nTop 3 Most Active Hours ({utc_name}):")
        hourly_avg_utc = df_rate.groupby('Hour_UTC')['Rate_Per_Hour'].mean().sort_values(ascending=False)
        for hour, rate in hourly_avg_utc.head(3).items():
            print(f"  - {hour:02d}:00 - Average rate: {rate:,.2f}/hour")

        print(f"\nTop 3 Most Active Hours ({local_name}):")
        hourly_avg_local = df_rate.groupby('Hour_Local')['Rate_Per_Hour'].mean().sort_values(ascending=False)
        for hour, rate in hourly_avg_local.head(3).items():
            print(f"  - {hour:02d}:00 - Average rate: {rate:,.2f}/hour")

    print("="*60 + "\n")

def main(csv_filepath, output_dir):
    """Main function to run all visualizations"""
    # Load and process data
    print("Loading and processing data...")
    df, utc_name, local_name = load_and_process_data(csv_filepath)

    # Generate summary statistics
    generate_summary_stats(df, utc_name, local_name)

    # Create visualizations
    print("Creating visualizations...")

    # Plot 1: Raw values over time (UTC, linear)
    fig1 = plot_raw_values(df, utc_name)
    save_path1 = os.path.join(output_dir, 'raw_values_plot.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {save_path1}")

    # Plot 2: Raw values with log scale (UTC only)
    fig2 = plot_raw_values_log(df, utc_name)
    save_path2 = os.path.join(output_dir, 'raw_values_log_plot.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {save_path2}")

    # Plot 3: Recent raw values (last 30 days, UTC + local)
    fig3a = plot_recent_raw_values(df, utc_name, local_name)
    save_path3a = os.path.join(output_dir, 'recent_raw_values_plot.png')
    plt.savefig(save_path3a, dpi=300, bbox_inches='tight')
    plt.close(fig3a)
    print(f"Saved: {save_path3a}")

    # Plot 3: Hourly rate analysis
    fig3 = plot_hourly_rate_analysis(df, utc_name, local_name)
    save_path3 = os.path.join(output_dir, 'hourly_rate_analysis.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3) # Close figure
    print(f"Saved: {save_path3}")

    # Plot 4: Daily activity heatmaps
    fig4 = plot_activity_heatmaps(df, utc_name, local_name)
    save_path4 = os.path.join(output_dir, 'daily_activity_heatmap.png')
    plt.savefig(save_path4, dpi=300, bbox_inches='tight')
    plt.close(fig4) # Close figure
    print(f"Saved: {save_path4}")

    # Plot 5: Day of week heatmaps
    fig5 = plot_day_of_week_heatmaps(df, utc_name, local_name)
    save_path5 = os.path.join(output_dir, 'day_of_week_heatmap.png')
    plt.savefig(save_path5, dpi=300, bbox_inches='tight')
    plt.close(fig5) # Close figure
    print(f"Saved: {save_path5}")


# Example usage
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set the directory where your CSV file is located and where plot images will be saved.
    # IMPORTANT: Use a raw string (r"...") on Windows or forward slashes ("/") for compatibility.
    # Example for Windows: DATA_DIRECTORY = r"C:\Users\YourUser\Documents\Radio"
    DATA_DIRECTORY = "/home/e/Documents/Radio/qrzLookups"
    # --- END CONFIGURATION ---

    # Ensure the output directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

    # Define the input CSV filename
    csv_filename = "NR8E_QRZ_stats.csv"
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(DATA_DIRECTORY, csv_filename)

    try:
        # Pass the directory to the main function for saving files
        main(csv_file_path, DATA_DIRECTORY)
        print(f"\nVisualization complete! Check the generated PNG files in '{DATA_DIRECTORY}'.")
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please ensure the CSV file is in the specified directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check that your CSV has 'Time' and 'Hits' columns.")
