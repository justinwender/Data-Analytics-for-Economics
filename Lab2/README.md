## 📉 The Illusion of Growth & the Composition Effect  
**Deflating History with FRED**

### Objective
To analyze long-run wage dynamics in the United States by correcting nominal wage data for inflation and labor-market composition effects, with a focus on distinguishing real income growth from statistical artifacts.

---

### Methodology
- Built a reproducible **Python data pipeline** to ingest live macroeconomic time series directly from the **Federal Reserve Economic Data (FRED) API**.
- Retrieved nominal average hourly earnings data and Consumer Price Index (CPI) measures to construct a historical series of **real wages**, adjusting for inflation.
- Identified a pronounced anomaly in nominal and real wage growth during **2020**, coinciding with the COVID-19 pandemic.
- Addressed this anomaly by incorporating the **Employment Cost Index (ECI)**, which controls for shifts in workforce composition and provides a cleaner measure of underlying labor cost growth.
- Compared real wage series derived from headline earnings data versus ECI-adjusted measures to isolate the impact of **composition bias**.

---

### Key Findings
The analysis illustrates a persistent **“money illusion”** in wage reporting: despite nominal wage growth over the past five decades, real wages remain largely flat when adjusted for inflation. This finding is consistent with broader concerns about long-run wage stagnation.

Additionally, the apparent wage surge observed in 2020 is shown to be a **statistical artifact rather than a true increase in labor demand**. By controlling for labor-force composition using the Employment Cost Index, the analysis demonstrates that the spike was driven primarily by the disproportionate exit of low-wage workers during the pandemic — a phenomenon known as the **composition effect**.

This “pandemic paradox” highlights the importance of careful variable selection and economic context when interpreting headline labor market statistics, particularly during periods of extreme disruption.
