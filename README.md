# QRZ Lookup Tracker

Tracks the cumulative lookup count on a QRZ callsign page over time and generates visualizations by Evan, NR8E.

## How it works

`qrzLookupTracker.sh` runs hourly (or as often as you'd like) via cron. It authenticates to QRZ using a session cookie, verifies the session is valid (to avoid inflating the count with unauthenticated visits), records the current lookup count to a CSV, then regenerates all plots via `qrzHitsViz.py`.

If the session expires, a `.session_invalid` sentinel is created and all further fetches are halted until the token is refreshed.

## Setup

1. Copy `.secrets.example` to `.secrets` and fill in your values:
   ```
   QRZ_SESSION_TOKEN=your_xf_session_token_here
   QRZ_CALLSIGN=your_callsign_here
   ```
   The session token is the value of the `xf_session` cookie from a logged-in QRZ browser session.

   To get that, go to [QRZ](https://www.qrz.com) and make sure you're logged in. Then, use your browser's web developer tools to view your storage for the page. Under that, there's probably a "cookies" section. Open the qrz.com one, and you'll see something that says "xf_session". There ya go! Copy that and use it as explained above.

3. Make the script executable:
   ```
   chmod +x qrzLookupTracker.sh
   ```

4. Add a cron entry:
   ```
   0 */1 * * * /path/to/qrzLookups/qrzLookupTracker.sh
   ```

## Files

| File | Description |
|---|---|
| `qrzLookupTracker.sh` | Main collection script |
| `qrzHitsViz.py` | Generates all plots from the CSV |
| `<CALLSIGN>_QRZ_stats.csv` | Collected data (timestamp, hit count) |
| `.secrets` | Credentials — **never commit this** |
| `.secrets.example` | Template for `.secrets` |
| `qrzTracker.log` | Log of script runs and errors |

## Plots generated

| File | Description |
|---|---|
| `raw_values_plot.png` | All-time lookup count, linear scale (UTC) |
| `raw_values_log_plot.png` | All-time lookup count, log scale (UTC) |
| `recent_raw_values_plot.png` | Last 30 days, linear scale (UTC + local time) |
| `hourly_rate_analysis.png` | Average lookup rate by hour of day |
| `daily_activity_heatmap.png` | Activity heatmap by date and hour |
| `day_of_week_heatmap.png` | Activity heatmap by day of week and hour |

## Session expiry

When the session expires, the script will:
1. Log an error to `qrzTracker.log`
2. Send a desktop notification
3. Create `.session_invalid` to halt further fetches

To resume: update `QRZ_SESSION_TOKEN` in `.secrets`, then delete `.session_invalid`.

## Thanks

Special thanks to Todd, KE2AEQ for helping out with the original shell script!
