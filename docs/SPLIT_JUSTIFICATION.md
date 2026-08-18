# Why the split is time-based (and global)

## The decision

Events are ordered by timestamp and cut at global quantiles: 85% train, 7.5%
validation, 7.5% test. No user, item, or interaction crosses a boundary in the
wrong direction.

## Why not a random split

A random split is a **correctness error** in a recommender, not a stylistic one.
Three distinct leaks:

1. **Future-to-past leakage inside a user.** If user *u*'s interaction at day 90
   is in train and their interaction at day 30 is in test, the model has seen the
   answer. Reported recall is inflated by an amount you cannot bound.
2. **Item-lifecycle leakage.** Item popularity is non-stationary. A random split
   lets the model learn that an item was popular in week 20 and then rewards it
   for "predicting" that popularity in week 3.
3. **Co-occurrence leakage through the item tower.** Even a *per-user* time split
   leaks: user A's future co-view of (X, Y) shapes the shared item embeddings
   that are used to score user B's past. This is why the split here is global,
   not per-user.

## Why not leave-one-out

Leave-one-out (hold out each user's last interaction) is common in papers and is
defensible for measuring next-item ranking. It is the wrong protocol *for this
system* because production retrains on a cadence and then serves a **window** of
future traffic, not one event per user. The offline protocol should mirror the
serving regime; a window split does, LOO does not.

## What the split costs, stated plainly

* **Cold entities.** Users and items that appear only after the cut are
  unreachable by a two-tower model. They are excluded from the metrics and the
  exclusion is printed with every run (`cold_start_report` in `data.py`), because
  a recall number that silently drops the hard cases is not comparable to one
  that doesn't.
* **Smaller effective eval set.** The test window is one slice of time, so
  seasonality is not averaged out. With 180 simulated days the test window is
  ~13 days. On real data I would report the metric across several rolling
  windows rather than one, and the code supports that by re-running with a
  different `train_end_q`.
* **Validation is used for early-stopping/config only.** Test is touched once per
  reported configuration. Every number in the README states which split it came
  from.

## What would change this decision

If the product were a cold-start-dominated catalogue (news, marketplace listings
with hours-long lifetimes), the right protocol is a *rolling* backtest with a
retrain at each step, because the question shifts from "does the model rank well"
to "does the pipeline keep up". That is a different experiment and it is listed
in the roadmap rather than claimed here.
