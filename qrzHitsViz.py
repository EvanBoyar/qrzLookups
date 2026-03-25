import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, AutoDateLocator, DayLocator, HourLocator
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import pytz
from tzlocal import get_localzone
import os

# Modern clean style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': '#cccccc',
    'grid.linestyle': '-',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'axes.titlesize': 14,
    'axes.titleweight': 'normal',
    'axes.labelsize': 11,
    'axes.labelcolor': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'text.color': '#333333',
})

# Consistent palette: primary, secondary, accent
COLOR_PRIMARY = '#3b82f6'    # blue
COLOR_SECONDARY = '#10b981'  # green
COLOR_FILL = '#3b82f6'
FILL_ALPHA = 0.08
CMAP_HEAT = 'YlGnBu'


def load_and_process_data(filepath):
    """Load CSV and process datetime columns for both UTC and local time"""
    df = pd.read_csv(filepath)

    df['Time_UTC'] = pd.to_datetime(df['Time'], utc=True)

    local_tz = get_localzone()
    df['Time_Local'] = df['Time_UTC'].dt.tz_convert(local_tz)

    df = df.sort_values('Time_UTC')

    df['Value_Change'] = df['Hits'].diff()
    df['Time_Delta'] = df['Time_UTC'].diff().dt.total_seconds() / 3600
    df['Rate_Per_Hour'] = df['Value_Change'] / df['Time_Delta']

    df['Hour_UTC'] = df['Time_UTC'].dt.hour
    df['Day_of_Week_UTC'] = df['Time_UTC'].dt.day_name()
    df['Date_UTC'] = df['Time_UTC'].dt.date

    df['Hour_Local'] = df['Time_Local'].dt.hour
    df['Day_of_Week_Local'] = df['Time_Local'].dt.day_name()
    df['Date_Local'] = df['Time_Local'].dt.date

    utc_name = "UTC"
    local_name = str(local_tz)

    return df, utc_name, local_name


def plot_raw_values(df, utc_name, figsize=(14, 5)):
    """Plot raw values over time in linear scale (UTC only)"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    use_markers = len(df) < 50
    ax.plot(df['Time_UTC'], df['Hits'], linewidth=2,
            marker='o' if use_markers else None,
            markersize=4 if use_markers else None,
            color=COLOR_PRIMARY, zorder=3)
    y_min = df['Hits'].min()
    y_pad = (df['Hits'].max() - y_min) * 0.05
    ax.fill_between(df['Time_UTC'], df['Hits'], y_min - y_pad, alpha=FILL_ALPHA, color=COLOR_FILL)
    ax.set_ylim(y_min - y_pad, None)

    ax.set_xlabel(f'Date/Time ({utc_name})')
    ax.set_ylabel('Hits')
    ax.set_title(f'QRZ Profile Hits Over Time ({utc_name})')
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=5, maxticks=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    if use_markers:
        for idx, row in df.iterrows():
            ax.annotate(f'{row["Hits"]:,}',
                        xy=(row['Time_UTC'], row['Hits']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    plt.tight_layout()
    return fig


def plot_recent_raw_values(df, utc_name, local_name, days=30, figsize=(14, 9)):
    """Plot the last N days of raw values for both UTC and local time"""
    cutoff = df['Time_UTC'].max() - pd.Timedelta(days=days)
    df_recent = df[df['Time_UTC'] >= cutoff]

    local_tz = get_localzone()

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Both plots use Time_UTC as x-data, but tick at midnight in their own timezone
    for ax, tz, color, label in [
        (axes[0], pytz.UTC,   COLOR_PRIMARY,   utc_name),
        (axes[1], local_tz,   COLOR_SECONDARY,  local_name),
    ]:
        use_markers = len(df_recent) < 50
        ax.plot(df_recent['Time_UTC'], df_recent['Hits'],
                linewidth=2, color=color, zorder=3,
                marker='o' if use_markers else None,
                markersize=4 if use_markers else None)
        r_min = df_recent['Hits'].min()
        r_pad = (df_recent['Hits'].max() - r_min) * 0.05
        ax.fill_between(df_recent['Time_UTC'], df_recent['Hits'], r_min - r_pad,
                        alpha=FILL_ALPHA, color=color)
        ax.set_ylim(r_min - r_pad, None)
        ax.set_xlabel(f'Date/Time ({label})')
        ax.set_ylabel('Hits')
        ax.set_title(f'Hits — Last {days} Days ({label})')
        ax.xaxis.set_major_locator(DayLocator(tz=tz))
        ax.xaxis.set_major_formatter(DateFormatter('%b %d', tz=tz))
        ax.xaxis.set_minor_locator(HourLocator(byhour=[0, 6, 12, 18], tz=tz))
        ax.tick_params(axis='x', which='minor', length=4, color='#aaaaaa')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    return fig


def plot_raw_values_log(df, utc_name, figsize=(14, 5)):
    """Plot raw values over time in logarithmic scale (UTC only)"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_min_exp = int(np.floor(np.log10(df['Hits'].min())))
    y_max_exp = int(np.ceil(np.log10(df['Hits'].max())))
    y_min = 10 ** y_min_exp
    y_max = 10 ** y_max_exp
    major_ticks = [10**i for i in range(y_min_exp, y_max_exp + 1)]
    minor_ticks = [m * 10**i for i in range(y_min_exp, y_max_exp) for m in range(2, 10)]

    use_markers = len(df) < 50
    ax.plot(df['Time_UTC'], df['Hits'], linewidth=2, color=COLOR_PRIMARY, zorder=3,
            marker='o' if use_markers else None,
            markersize=4 if use_markers else None)
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f'Date/Time ({utc_name})')
    ax.set_ylabel('Hits (Log Scale)')
    ax.set_title(f'QRZ Profile Hits Over Time — Log Scale ({utc_name})')
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", color="#bbbbbb", alpha=0.4)
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=5, maxticks=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f'$10^{{{int(np.log10(t))}}}$' for t in major_ticks])
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels([f'{t:,}' for t in minor_ticks], minor=True)
    ax.tick_params(axis='y', which='minor', labelsize=9, labelcolor='#555555')

    plt.tight_layout()
    return fig


def plot_hourly_rate_analysis(df, utc_name, local_name, figsize=(14, 5)):
    """Analyze average hourly rate of change for both UTC and local time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    df_rate = df.dropna(subset=['Rate_Per_Hour'])

    for ax, hour_col, color, label in [
        (ax1, 'Hour_UTC',   COLOR_PRIMARY,   utc_name),
        (ax2, 'Hour_Local', COLOR_SECONDARY, local_name),
    ]:
        if len(df_rate) == 0:
            continue

        hourly_stats = df_rate.groupby(hour_col)['Rate_Per_Hour'].agg(
            ['mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        )
        hourly_stats.columns = ['mean', 'q25', 'q75']
        hourly_stats = hourly_stats.reset_index()

        bars = ax.bar(hourly_stats[hour_col], hourly_stats['mean'],
                      alpha=0.85, color=color, edgecolor='white',
                      linewidth=0.5)

        ax.set_xlabel(f'Hour of Day ({label})')
        ax.set_ylabel('Mean Rate of Change (per hour)')
        ax.set_title(f'Hourly Rate of Change — {label}')
        ax.set_xticks(range(0, 24))
        ax.grid(True, alpha=0.2, axis='y')
        ax.axhline(y=0, color='#999999', linewidth=0.8, zorder=1)

    plt.tight_layout()
    return fig


def plot_activity_heatmaps(df, utc_name, local_name, figsize=(18, 7)):
    """Create activity heatmaps for both UTC and local time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, date_col, hour_col, label in [
        (ax1, 'Date_UTC',   'Hour_UTC',   utc_name),
        (ax2, 'Date_Local', 'Hour_Local', local_name),
    ]:
        if len(df[date_col].unique()) > 7:
            df_pivot = df.pivot_table(values='Value_Change',
                                      index=hour_col,
                                      columns=date_col,
                                      aggfunc='sum')

            if not df_pivot.empty:
                sns.heatmap(df_pivot, cmap=CMAP_HEAT, ax=ax,
                            cbar_kws={'label': 'Value Change', 'shrink': 0.8})
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Hour of Day ({label})')
                ax.set_title(f'Daily Activity Heatmap — {label}')

                # Thin out x-axis labels to prevent overlap
                n_dates = len(df_pivot.columns)
                step = max(1, n_dates // 12)
                ax.set_xticks(range(0, n_dates, step))
                ax.set_xticklabels(
                    [str(df_pivot.columns[i]) for i in range(0, n_dates, step)],
                    rotation=45, ha='right', fontsize=9
                )
        else:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap\n(need more than 7 days)',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'Daily Activity Heatmap — {label}')

    plt.tight_layout()
    return fig


def plot_day_of_week_heatmaps(df, utc_name, local_name, figsize=(16, 10)):
    """Create day-of-week activity heatmaps showing patterns by weekday"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for ax, dow_col, hour_col, label in [
        (ax1, 'Day_of_Week_UTC',   'Hour_UTC',   utc_name),
        (ax2, 'Day_of_Week_Local', 'Hour_Local', local_name),
    ]:
        df_dow = df.groupby([dow_col, hour_col])['Rate_Per_Hour'].mean().reset_index()
        if not df_dow.empty:
            df_pivot_dow = df_dow.pivot(index=dow_col, columns=hour_col, values='Rate_Per_Hour')
            df_pivot_dow = df_pivot_dow.reindex([d for d in day_order if d in df_pivot_dow.index])

            if not df_pivot_dow.empty:
                sns.heatmap(df_pivot_dow, cmap=CMAP_HEAT, ax=ax,
                            cbar_kws={'label': 'Avg Rate/Hour', 'shrink': 0.8},
                            annot=False, linewidths=0.5, linecolor='white')
                ax.set_xlabel(f'Hour of Day ({label})')
                ax.set_ylabel('Day of Week')
                ax.set_title(f'Average Activity by Day of Week — {label}')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
            ax.set_title(f'Average Activity by Day of Week — {label}')

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

    df_rate = df.dropna(subset=['Rate_Per_Hour'])
    if len(df_rate) > 0:
        print(f"\nRate of Change Statistics (per hour):")
        print(f"  - Mean Rate: {df_rate['Rate_Per_Hour'].mean():,.2f}")
        print(f"  - Median Rate: {df_rate['Rate_Per_Hour'].median():,.2f}")
        print(f"  - Max Rate: {df_rate['Rate_Per_Hour'].max():,.2f}")
        print(f"  - Min Rate: {df_rate['Rate_Per_Hour'].min():,.2f}")

        print(f"\nTop 3 Most Active Hours ({utc_name}):")
        hourly_avg_utc = df_rate.groupby('Hour_UTC')['Rate_Per_Hour'].mean().sort_values(ascending=False)
        for hour, rate in hourly_avg_utc.head(3).items():
            print(f"  - {hour:02d}:00 - Average rate: {rate:,.2f}/hour")

        print(f"\nTop 3 Most Active Hours ({local_name}):")
        hourly_avg_local = df_rate.groupby('Hour_Local')['Rate_Per_Hour'].mean().sort_values(ascending=False)
        for hour, rate in hourly_avg_local.head(3).items():
            print(f"  - {hour:02d}:00 - Average rate: {rate:,.2f}/hour")

    print("="*60 + "\n")


def plot_contribution_calendar(df, local_name, figsize=(16, 4)):
    """GitHub-style contribution calendar showing daily hit gains"""
    daily = df.groupby('Date_Local')['Hits'].agg(['first', 'last'])
    daily['gain'] = daily['last'] - daily['first']
    daily = daily['gain']

    if daily.empty:
        return None

    start_date = daily.index.min()
    end_date = daily.index.max()
    all_dates = pd.date_range(start_date, end_date, freq='D').date
    daily = daily.reindex(all_dates, fill_value=0)

    # Build week/day grid aligned so columns are weeks starting on Sunday
    first_dow = pd.Timestamp(all_dates[0]).dayofweek  # Mon=0..Sun=6
    # Shift to Sun=0 convention
    first_dow_sun = (first_dow + 1) % 7
    n_days = len(daily) + first_dow_sun
    n_weeks = int(np.ceil(n_days / 7))

    grid = np.full((7, n_weeks), np.nan)
    for i, (date, val) in enumerate(daily.items()):
        col = (i + first_dow_sun) // 7
        row = (i + first_dow_sun) % 7
        grid[row, col] = val

    vmax = max(daily.quantile(0.95), 1)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps['Greens'].copy()
    cmap.set_bad(color='#ebedf0')
    # Zero days get lightest shade
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    for row in range(7):
        for col in range(n_weeks):
            val = grid[row, col]
            if np.isnan(val):
                color = '#ebedf0'
            elif val <= 0:
                color = '#ebedf0'
            else:
                color = cmap(norm(min(val, vmax)))
            rect = mpatches.FancyBboxPatch(
                (col * 1.15, (6 - row) * 1.15), 1, 1,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, n_weeks * 1.15 + 0.5)
    ax.set_ylim(-0.5, 7 * 1.15 + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    for i, label in enumerate(day_labels):
        if i % 2 == 1:
            ax.text(-0.8, i * 1.15 + 0.5, label, ha='right', va='center',
                    fontsize=9, color='#555555')

    # Month labels along the top
    months_shown = set()
    for i, date in enumerate(daily.index):
        ts = pd.Timestamp(date)
        col = (i + first_dow_sun) // 7
        month_key = (ts.year, ts.month)
        if month_key not in months_shown and ts.day <= 7:
            months_shown.add(month_key)
            ax.text(col * 1.15, -0.8, ts.strftime('%b'), ha='left', va='center',
                    fontsize=9, color='#555555')

    ax.axis('off')
    ax.set_title(f'Daily Hit Gains — Contribution Calendar ({local_name})',
                 fontsize=14, pad=15, color='#333333')

    # Legend
    legend_x = n_weeks * 1.15 - 6
    legend_y = 7 * 1.15 + 1.5
    ax.text(legend_x - 1.5, legend_y + 0.5, 'Less', fontsize=8, color='#555555',
            ha='right', va='center')
    for j, frac in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        color = '#ebedf0' if frac == 0 else cmap(frac)
        rect = mpatches.FancyBboxPatch(
            (legend_x + j * 1.15, legend_y), 1, 1,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
    ax.text(legend_x + 5 * 1.15 + 0.5, legend_y + 0.5, 'More', fontsize=8,
            color='#555555', ha='left', va='center')

    plt.tight_layout()
    return fig


def plot_polar_clock(df, utc_name, local_name, figsize=(12, 6)):
    """Polar/clock plot showing hit rate by hour of day"""
    df_rate = df.dropna(subset=['Rate_Per_Hour'])
    if df_rate.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    subplot_kw={'projection': 'polar'})

    for ax, hour_col, color, label in [
        (ax1, 'Hour_UTC',   COLOR_PRIMARY,   utc_name),
        (ax2, 'Hour_Local', COLOR_SECONDARY, local_name),
    ]:
        hourly_avg = df_rate.groupby(hour_col)['Rate_Per_Hour'].mean()
        hourly_avg = hourly_avg.reindex(range(24), fill_value=0)

        # Angles: 0 = top (midnight), clockwise
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        values = hourly_avg.values
        # Close the polygon
        angles_closed = np.append(angles, angles[0])
        values_closed = np.append(values, values[0])

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        ax.fill(angles_closed, values_closed, alpha=0.2, color=color)
        ax.plot(angles_closed, values_closed, linewidth=2, color=color)

        ax.set_xticks(angles)
        ax.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=8, color='#555555')
        ax.set_title(f'Hourly Profile ({label})', pad=20, fontsize=12, color='#333333')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_anomaly_detection(df, utc_name, sigma=2, figsize=(14, 7)):
    """Detect and highlight spikes and quiet periods in hit rate"""
    df_rate = df.dropna(subset=['Rate_Per_Hour']).copy()
    if df_rate.empty:
        return None

    mean_rate = df_rate['Rate_Per_Hour'].mean()
    std_rate = df_rate['Rate_Per_Hour'].std()
    upper_thresh = mean_rate + sigma * std_rate
    lower_thresh = max(mean_rate - sigma * std_rate, 0)

    df_rate['is_spike'] = df_rate['Rate_Per_Hour'] > upper_thresh
    df_rate['is_quiet'] = df_rate['Rate_Per_Hour'] <= lower_thresh

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Top: rate over time with anomalies highlighted
    ax1.plot(df_rate['Time_UTC'], df_rate['Rate_Per_Hour'],
             linewidth=1, color='#999999', alpha=0.6, zorder=2)
    spikes = df_rate[df_rate['is_spike']]
    quiets = df_rate[df_rate['is_quiet']]
    ax1.scatter(spikes['Time_UTC'], spikes['Rate_Per_Hour'],
                color='#ef4444', s=30, zorder=4, label=f'Spike (>{sigma}\u03c3)')
    ax1.scatter(quiets['Time_UTC'], quiets['Rate_Per_Hour'],
                color='#6366f1', s=20, marker='v', zorder=3, label=f'Quiet (\u2264{sigma}\u03c3 below)')
    ax1.axhline(y=mean_rate, color=COLOR_PRIMARY, linewidth=1, linestyle='--',
                alpha=0.7, label=f'Mean ({mean_rate:.1f}/hr)')
    ax1.axhline(y=upper_thresh, color='#ef4444', linewidth=0.8, linestyle=':',
                alpha=0.5, label=f'+{sigma}\u03c3 ({upper_thresh:.1f}/hr)')
    if lower_thresh > 0:
        ax1.axhline(y=lower_thresh, color='#6366f1', linewidth=0.8, linestyle=':',
                    alpha=0.5, label=f'-{sigma}\u03c3 ({lower_thresh:.1f}/hr)')
    ax1.set_xlabel(f'Date/Time ({utc_name})')
    ax1.set_ylabel('Rate (hits/hour)')
    ax1.set_title(f'Anomaly Detection — Rate of Change ({utc_name})')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.xaxis.set_major_formatter(DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(AutoDateLocator(minticks=5, maxticks=12))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Bottom: daily gains with spike days highlighted
    daily = df.groupby('Date_UTC')['Hits'].agg(['first', 'last'])
    daily['gain'] = daily['last'] - daily['first']
    daily_mean = daily['gain'].mean()
    daily_std = daily['gain'].std()
    daily_upper = daily_mean + sigma * daily_std

    colors = ['#ef4444' if g > daily_upper else COLOR_PRIMARY for g in daily['gain']]
    ax2.bar(range(len(daily)), daily['gain'], color=colors, alpha=0.8, width=1.0)
    ax2.axhline(y=daily_mean, color='#333333', linewidth=1, linestyle='--',
                alpha=0.5, label=f'Mean ({daily_mean:.1f}/day)')
    ax2.axhline(y=daily_upper, color='#ef4444', linewidth=0.8, linestyle=':',
                alpha=0.5, label=f'+{sigma}\u03c3 ({daily_upper:.1f}/day)')

    # X-axis labels for daily bar chart
    n_days = len(daily)
    step = max(1, n_days // 15)
    ax2.set_xticks(range(0, n_days, step))
    ax2.set_xticklabels([str(daily.index[i]) for i in range(0, n_days, step)],
                        rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Hits Gained')
    ax2.set_title('Daily Gains with Spike Days Highlighted')
    ax2.legend(fontsize=9, loc='upper right')

    # Print spike summary
    spike_days = daily[daily['gain'] > daily_upper]
    if len(spike_days) > 0:
        print(f"\nSpike days (>{sigma}\u03c3 = >{daily_upper:.0f} hits/day):")
        for date, row in spike_days.sort_values('gain', ascending=False).iterrows():
            print(f"  {date}: +{row['gain']:.0f} hits")

    quiet_days = daily[daily['gain'] <= 0]
    if len(quiet_days) > 0:
        print(f"\nZero/negative growth days: {len(quiet_days)}")
        if len(quiet_days) <= 10:
            for date, row in quiet_days.iterrows():
                print(f"  {date}: {row['gain']:+.0f} hits")

    plt.tight_layout()
    return fig


def plot_milestone_forecast(df, utc_name, figsize=(14, 6)):
    """Project when future milestones will be reached"""
    df_clean = df.dropna(subset=['Hits']).copy()
    if df_clean.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual data
    ax.plot(df_clean['Time_UTC'], df_clean['Hits'], linewidth=2, color=COLOR_PRIMARY,
            label='Actual', zorder=3)

    current_hits = df_clean['Hits'].iloc[-1]
    current_time = df_clean['Time_UTC'].iloc[-1]
    start_time = df_clean['Time_UTC'].iloc[0]

    # Linear regression on the full dataset
    x_numeric = (df_clean['Time_UTC'] - start_time).dt.total_seconds().values
    y = df_clean['Hits'].values
    coeffs = np.polyfit(x_numeric, y, 1)
    slope_per_sec = coeffs[0]
    intercept = coeffs[1]

    # Also compute recent trend (last 30 days)
    cutoff_30 = current_time - pd.Timedelta(days=30)
    df_recent = df_clean[df_clean['Time_UTC'] >= cutoff_30]
    if len(df_recent) > 10:
        x_recent = (df_recent['Time_UTC'] - start_time).dt.total_seconds().values
        y_recent = df_recent['Hits'].values
        coeffs_recent = np.polyfit(x_recent, y_recent, 1)
        slope_recent = coeffs_recent[0]
    else:
        slope_recent = slope_per_sec

    # Generate milestones
    step = 500
    next_milestone = int(np.ceil(current_hits / step) * step)
    milestones = [next_milestone + i * step for i in range(5)]
    milestones = [m for m in milestones if m > current_hits][:5]

    # Project future dates (extend 2x the current data span)
    total_span_sec = x_numeric[-1]
    future_span = total_span_sec * 2
    future_x = np.linspace(0, x_numeric[-1] + future_span, 500)
    future_times = [start_time + pd.Timedelta(seconds=float(s)) for s in future_x]

    # Linear trend lines
    y_linear = slope_per_sec * future_x + intercept
    y_recent_trend = slope_recent * future_x + (coeffs_recent[1] if len(df_recent) > 10 else intercept)

    ax.plot(future_times, y_linear, '--', color=COLOR_PRIMARY, alpha=0.5,
            linewidth=1.5, label=f'Overall trend ({slope_per_sec * 86400:.1f}/day)')
    ax.plot(future_times, y_recent_trend, '--', color=COLOR_SECONDARY, alpha=0.5,
            linewidth=1.5, label=f'Recent 30-day trend ({slope_recent * 86400:.1f}/day)')

    # Mark milestones on both trend lines
    print(f"\nMilestone Projections:")
    print(f"  Current: {current_hits:,} hits as of {current_time.strftime('%Y-%m-%d')}")
    print(f"  Overall trend: {slope_per_sec * 86400:.1f} hits/day")
    print(f"  Recent 30-day trend: {slope_recent * 86400:.1f} hits/day")
    print()

    for milestone in milestones:
        # Overall trend ETA
        if slope_per_sec > 0:
            sec_to_milestone = (milestone - intercept) / slope_per_sec
            eta_overall = start_time + pd.Timedelta(seconds=float(sec_to_milestone))
        else:
            eta_overall = None

        # Recent trend ETA
        if slope_recent > 0:
            sec_to_milestone_r = (milestone - coeffs_recent[1]) / slope_recent if len(df_recent) > 10 else None
            eta_recent = start_time + pd.Timedelta(seconds=float(sec_to_milestone_r)) if sec_to_milestone_r else None
        else:
            eta_recent = None

        # Draw milestone line
        ax.axhline(y=milestone, color='#999999', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.text(future_times[-1], milestone, f'  {milestone:,}',
                va='bottom', ha='right', fontsize=9, color='#555555')

        overall_str = eta_overall.strftime('%Y-%m-%d') if eta_overall and eta_overall < future_times[-1] else '—'
        recent_str = eta_recent.strftime('%Y-%m-%d') if eta_recent and eta_recent < future_times[-1] else '—'
        print(f"  {milestone:,} hits — overall: {overall_str}, recent trend: {recent_str}")

    ax.set_xlabel(f'Date ({utc_name})')
    ax.set_ylabel('Hits')
    ax.set_title('Milestone Forecast')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=5, maxticks=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    return fig


def main(csv_filepath, output_dir):
    """Main function to run all visualizations"""
    print("Loading and processing data...")
    df, utc_name, local_name = load_and_process_data(csv_filepath)

    generate_summary_stats(df, utc_name, local_name)

    print("Creating visualizations...")

    fig1 = plot_raw_values(df, utc_name)
    save_path1 = os.path.join(output_dir, 'raw_values_plot.png')
    plt.savefig(save_path1, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {save_path1}")

    fig2 = plot_raw_values_log(df, utc_name)
    save_path2 = os.path.join(output_dir, 'raw_values_log_plot.png')
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {save_path2}")

    fig3a = plot_recent_raw_values(df, utc_name, local_name)
    save_path3a = os.path.join(output_dir, 'recent_raw_values_plot.png')
    plt.savefig(save_path3a, dpi=200, bbox_inches='tight')
    plt.close(fig3a)
    print(f"Saved: {save_path3a}")

    fig3 = plot_hourly_rate_analysis(df, utc_name, local_name)
    save_path3 = os.path.join(output_dir, 'hourly_rate_analysis.png')
    plt.savefig(save_path3, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {save_path3}")

    fig4 = plot_activity_heatmaps(df, utc_name, local_name)
    save_path4 = os.path.join(output_dir, 'daily_activity_heatmap.png')
    plt.savefig(save_path4, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {save_path4}")

    fig5 = plot_day_of_week_heatmaps(df, utc_name, local_name)
    save_path5 = os.path.join(output_dir, 'day_of_week_heatmap.png')
    plt.savefig(save_path5, dpi=200, bbox_inches='tight')
    plt.close(fig5)
    print(f"Saved: {save_path5}")

    fig6 = plot_contribution_calendar(df, local_name)
    if fig6:
        save_path6 = os.path.join(output_dir, 'contribution_calendar.png')
        plt.savefig(save_path6, dpi=200, bbox_inches='tight')
        plt.close(fig6)
        print(f"Saved: {save_path6}")

    fig7 = plot_polar_clock(df, utc_name, local_name)
    if fig7:
        save_path7 = os.path.join(output_dir, 'polar_clock.png')
        plt.savefig(save_path7, dpi=200, bbox_inches='tight')
        plt.close(fig7)
        print(f"Saved: {save_path7}")

    fig8 = plot_anomaly_detection(df, utc_name)
    if fig8:
        save_path8 = os.path.join(output_dir, 'anomaly_detection.png')
        plt.savefig(save_path8, dpi=200, bbox_inches='tight')
        plt.close(fig8)
        print(f"Saved: {save_path8}")

    fig9 = plot_milestone_forecast(df, utc_name)
    if fig9:
        save_path9 = os.path.join(output_dir, 'milestone_forecast.png')
        plt.savefig(save_path9, dpi=200, bbox_inches='tight')
        plt.close(fig9)
        print(f"Saved: {save_path9}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    csv_filename = "NR8E_QRZ_stats.csv"
    csv_file_path = os.path.join(SCRIPT_DIR, csv_filename)

    plot_dir = os.path.join(SCRIPT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    try:
        main(csv_file_path, plot_dir)
        print(f"\nVisualization complete! Check the generated PNG files in '{plot_dir}'.")
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please ensure the CSV file is in the specified directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check that your CSV has 'Time' and 'Hits' columns.")
