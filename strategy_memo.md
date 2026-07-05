# Network Operations Strategy Memo
### Closing the gap between promised and actual delivery times

**To:** Head of Network Operations
**From:** Data Science — Network Intelligence
**Re:** Graph-based ETA accuracy, bottleneck hubs, and where to invest next
**Length:** 2 pages

---

## The problem, in one line

Our ETA engine (OSRM) assumes clean roads and shortest paths. The network doesn't behave that
way. Across the trips we analysed, **the typical leg takes about twice as long as OSRM
predicts** (median actual-to-estimate ratio ≈ 2.0), and **~96% of corridors run more than 20%
slower than promised**. When the ETA is wrong, SLAs slip, customers churn, and capacity
planning downstream is built on numbers that were never going to hold.

This memo does two things: (1) shows we can predict ETAs far more accurately by treating the
network as a connected graph rather than a list of independent routes, and (2) names the
specific hubs where a fix returns the most.

---

## What changed: a graph beats the status quo

We rebuilt ETA prediction to use **network structure** — each facility's position in the graph
(how central it is, how connected, who it routes to) — on top of the usual distance and
route-type inputs. Same data, same test set, measured head-to-head:

| ETA method | Within 15% of actual | Avg. error (MAE) |
|---|---|---|
| OSRM (today) | **~5%** of legs | 107 |
| ML baseline (no graph) | ~35% of legs | 42 |
| **Graph-enhanced model** | **~48% of legs** | **32** |

The graph features alone cut average error by **~24%** and **nearly tripled** the share of legs
predicted within 15% of reality versus today's OSRM. This is a measured result on held-out
trips, not a projection. **Operational meaning:** planners get an ETA they can actually staff
and promise against on roughly half of all legs, instead of one in twenty.

---

## Where the delay concentrates: the top 5 bottleneck hubs

We ranked every facility by **SLA-breach contribution** — how much late time it generates,
combining traffic volume with how badly its corridors overshoot. A handful of hubs dominate.

| # | Hub | Why it matters |
|---|---|---|
| 1 | **Bhiwandi_Mankoli_HB** (Maharashtra) | Highest realised late-time in the network; dense in/out corridors, little bypass capacity. |
| 2 | **Gurgaon_Bilaspur_HB** (Haryana) | Most structurally central hub (highest betweenness) — delays here ripple network-wide. |
| 3 | **Kolkata_Dankuni_HB** (West Bengal) | Major eastern chokepoint with chronic overshoot. |
| 4 | **Kanpur_Central_H** (Uttar Pradesh) | High-volume northern transfer point running consistently late. |
| 5 | **Bangalore_Nelamangala_H** (Karnataka) | Central southern hub; high betweenness and degree. |

The **top 3 hubs alone account for ~7% of all SLA-breach in the network** — a disproportionate
return for three targeted interventions.

---

## Recommended interventions (corridor-specific)

1. **Bhiwandi_Mankoli_HB — facility upgrade (capacity/dwell).** Its delay is realised, not
   structural — i.e. shipments physically sit. Add sortation/dock capacity to cut dwell time.
2. **Gurgaon_Bilaspur_HB — parallel route / load-balancing.** This hub is the network's biggest
   structural chokepoint. Standing up an alternate corridor for its busiest lanes reduces
   single-point ripple risk more than raw capacity would.
3. **Kolkata_Dankuni_HB — facility upgrade + peak-hour shift.** Combine added capacity with
   moving flexible volume out of the worst congestion windows.
4. **Route-type shift (network-wide).** Carting carries a higher delay ratio and only wins on
   short feeder legs. **Move medium/long legs out of high-centrality hubs to FTL**, where the
   time and SLA gains justify the truck; keep Carting for short, low-criticality feeders.

---

## The prize: quantified impact

Modelling a facility upgrade at the **top 3 hubs** as removing ~60% of their excess delay
(re-runnable with your own uplift assumption):

- **~4% reduction in network-wide late delivery time** from just three sites.
- On a revenue-at-risk proxy of ~₹3.0M tied to chronically-late shipments in this sample,
  roughly **₹0.13M recovered** — and that scales with the full shipment base and a higher,
  realistic per-shipment SLA-penalty figure.
- Better ETAs everywhere: **~48% of legs within 15% of actual**, versus ~5% today, sharpening
  every downstream capacity and staffing decision.

---

## Recommended next steps

1. Greenlight upgrades at the **top 3 hubs**, starting with Bhiwandi_Mankoli_HB.
2. Pilot the **FTL-vs-Carting rule** on the highest-breach corridors out of central hubs.
3. Deploy the **graph-enhanced ETA model** into planning for the affected corridors and track
   realised vs predicted weekly.

*Supporting analysis, code, and visualizations: `delhivery_graph_eta.ipynb`. All impact figures
use explicit, adjustable assumptions documented in the notebook.*
