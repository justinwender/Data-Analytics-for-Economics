## 📊 The Cost of Living Crisis: A Data-Driven Analysis

### The Problem: Why the "Average" CPI Fails Students

The Consumer Price Index (CPI) is the most widely cited measure of inflation in the United States — yet it tells an incomplete story. The official CPI is constructed using a market basket weighted toward the consumption patterns of a typical American household: homeownership costs, vehicle purchases, healthcare expenditures, and other categories that do not reflect the financial reality of college students.

For students, the true cost of living is dominated by a fundamentally different basket: **tuition, rent, and food**. When these categories experience disproportionate price increases, the headline inflation rate systematically understates the erosion of student purchasing power. This project investigates whether a custom Student Price Index (SPI) diverges meaningfully from the official CPI — and by how much.

---

### Methodology: Python, APIs, and Index Theory

This analysis employs a **Laspeyres price index** approach — the same methodology underlying the official CPI — but with expenditure weights calibrated to student consumption patterns.

**Data Pipeline:**
- Built a reproducible Python pipeline using the **FRED API** (`fredapi`) to ingest official Bureau of Labor Statistics price indices
- Retrieved component indices for: College Tuition & Fees (`CUSR0000SEEB`), Rent of Primary Residence (`CUSR0000SEHA`), Admission to Entertainment (`CUSR0000SERA02`), and Food Away from Home (`CUSR0000SEFV`)
- Incorporated the Boston-Cambridge-Newton regional CPI (`CUURA103SA0`) to capture geographic cost variation in a major university market

**Index Construction:**
- Normalized all series to a common base year (2016 = 100) to enable valid cross-index comparison
- Constructed a weighted Student Price Index using the following allocation:
  - **Tuition**: 40%
  - **Rent**: 30%
  - **Entertainment**: 15%
  - **Prepared Food**: 15%

**Validation:**
- Cross-referenced FRED data with manually collected observational prices (2016–2024) for items in a typical student basket: tuition, one-bedroom rent, Chipotle burritos, and Maruchan ramen

---

### Key Findings

**My analysis reveals a 91 percentage point cumulative divergence between Student Costs and National Inflation since 1992.**

When both indices are normalized to 2016 = 100:
- The **National CPI** stood at 58.2 in January 1992, implying ~72% cumulative inflation from 1992 to 2016
- The **Student Price Index** stood at 38.0 in January 1992, implying ~163% cumulative inflation over the same period

This means student-weighted costs inflated at **more than twice the rate** of the general economy over a 24-year period.

**Component-Level Analysis (Observational Data, 2016–2024):**

| Item | 2016 Price | 2024 Price | Inflation |
|------|-----------|-----------|-----------|
| Tuition | $45,000 | $58,000 | **28.9%** |
| Rent (1-Bed) | $1,200/mo | $1,800/mo | **50.0%** |
| Chipotle Burrito | $7.50 | $11.50 | **53.3%** |
| Maruchan Ramen | $0.25 | $0.50 | **100.0%** |

The official CPI increased approximately 25–30% over the same period — yet every item in the student basket exceeded this benchmark, with staple foods experiencing the most severe relative inflation.

**Regional Disparity:**
The Boston-Cambridge-Newton CPI closely tracks national inflation but reveals that students in major university markets face compounding geographic cost premiums on top of sector-specific inflation.

---

### Implications

These findings have direct policy relevance:
1. **Financial Aid Adjustments**: Cost-of-attendance calculations that rely on headline CPI may systematically underestimate student need
2. **Wage Stagnation Context**: Entry-level wages benchmarked to general inflation fail to maintain student purchasing power
3. **Index Composition Matters**: The choice of basket weights is not merely technical — it determines whose economic reality gets measured

The divergence between official inflation statistics and lived experience is not a failure of measurement, but a reflection of whose consumption patterns are deemed representative.

---

### Technical Stack
- **Language**: Python 3
- **Data Source**: Federal Reserve Economic Data (FRED) API
- **Libraries**: `fredapi`, `pandas`, `matplotlib`
- **Index Methodology**: Laspeyres fixed-weight price index
