# NYC Congestion Pricing: Difference-in-Differences Analysis
 
## What this is
 
I wanted to know if NYC's congestion pricing program actually reduced traffic, or if any drop was just normal noise that would've happened anyway. To answer that I used a difference-in-differences (DiD) setup on MTA bridge and tunnel crossing data, comparing the tolled crossings to the untolled ones before and after tolling started.
 
Short answer: traffic at the tolled crossings dropped by about 3.4%, and that result held up even after I went back and fixed some problems with my first pass at the analysis.
 
## The data
 
I used the MTA Bridges and Tunnels Hourly Crossings dataset, aggregated to a daily facility-level panel from January 2023 through 2026. The treated facilities are the Queens Midtown Tunnel and the Hugh L. Carey Tunnel, since those are the ones that got tolled starting January 5, 2025. Everything else in the dataset is the control group.
 
## First pass
 
My first model was a straightforward OLS regression: log traffic on treatment status, a post-period dummy, the interaction between the two, and day-of-week fixed effects. I clustered the standard errors by month since I figured traffic data would have some kind of month-level pattern.
 
```
log_traffic ~ treated + post + treated:post + C(dow)
```
 
That gave me a treated x post coefficient of -0.0337, so about a 3.3% drop. I also went back and excluded some obviously weird days (a couple of major snowstorms, July 4th, Christmas) and the result barely moved.
 
## The problem I found
 
I ran a Durbin-Watson test on the residuals just to check my assumptions, and got values around 0.66 to 0.75 for both groups. Since 2.0 is what you'd expect with no autocorrelation, this told me there was a real day-to-day pattern left in the errors that my model wasn't accounting for. Clustering by month doesn't really help here because the correlation I was seeing was day-to-day, not month-to-month, so it was basically the wrong tool for the problem.
 
## Fixing it
 
To deal with this, I rebuilt the data as a panel indexed by entity (treated or control) and date, then refit it using PanelOLS with entity and time fixed effects instead of my original dummy variables. The time fixed effects do a better job than my manual outlier removal too, since they absorb anything that happened on a given date across both groups (weather, holidays, whatever), instead of me having to guess which days were weird.
 
For the standard errors, I switched to Driscoll-Kraay, which is built to handle both serial correlation and correlation between the treated and control groups on the same date. This gave me the same coefficient as before, -0.0337, which was a good sign that the original result wasn't just an artifact of a shaky model.
 
Driscoll-Kraay also depends on a bandwidth parameter (how many days back it looks when correcting for correlation), so I tested it at 4, 7, 14, and 21 days to make sure I wasn't just getting lucky with one setting:
 
| Bandwidth | Coefficient | Std. Error | P-value |
|-----------|-------------|------------|---------|
| 4         | -0.0337     | 0.0076     | 0.0000  |
| 7         | -0.0337     | 0.0087     | 0.0001  |
| 14        | -0.0337     | 0.0103     | 0.0012  |
| 21        | -0.0337     | 0.0112     | 0.0028  |
 
The coefficient didn't budge at all, and even at the widest window the result was still significant.
 
## What I found
 
Congestion pricing cut traffic at the tolled crossings by roughly 3.4% compared to the untolled ones. That number held up after I fixed the autocorrelation issue, tried different fixes, and tested a handful of parameter choices, so I'm fairly confident it's a real effect.
 
## Limitations
 
- Only two crossings are tolled in this dataset, so I can't say for sure this would generalize if congestion pricing expanded elsewhere
- This only looks at raw crossing counts, not trip purpose, vehicle type, or whether people just switched to transit
- I didn't look at longer-term effects like people slowly rerouting to untolled crossings over time
## Tools
 
Python, pandas, statsmodels, linearmodels (PanelOLS, Driscoll-Kraay), matplotlib