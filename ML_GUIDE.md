# Machine Learning Guide: Lessons from the Rookie Ranker

This guide covers the ML concepts behind this project. Every idea is explained through the lens of what we actually built.

---

## 1. What Kind of Problem Is This?

Machine learning problems fall into a few broad categories. Ours is a **regression** problem — we're predicting a continuous number (fantasy points) rather than a yes/no answer or a category.

Compare:
- *Classification*: "Will this player make the Pro Bowl?" (yes/no)
- *Regression*: "How many fantasy points will this player score?" (a number)

Regression is harder to evaluate because there's no clean right/wrong — there's only how close you got.

---

## 2. Supervised Learning

Our model is **supervised**, meaning it learns from examples where we already know the answer.

The setup:
- **Input (features)**: what we know about a player going into their rookie year — college stats, position, conference
- **Label (target)**: what actually happened — their NFL fantasy points that season

The model reads thousands of these examples and finds patterns: "players with X rushing yards and Y touchdowns in college tend to score Z fantasy points as rookies." Then it applies those patterns to new players we haven't seen yet.

This is why historical data matters so much. Without examples of what *actually happened*, the model has nothing to learn from.

---

## 3. The Training Pipeline

Before any learning happens, raw data goes through several transformations:

**Raw data → Processed data → Features → Model**

In our case:
1. We fetch college stats for players (raw)
2. We fetch NFL rookie stats for those same players (raw)
3. We match them up by name using fuzzy matching (processed)
4. We clean and select the relevant columns (features)
5. The model trains on those features

Each step is a place where things can go wrong. Bad matching in step 3 means the model learns from incorrect examples. Noisy features in step 4 means the model learns from noise instead of signal. A lot of ML work is just cleaning up these steps.

---

## 4. Features

A **feature** is any piece of information you give the model as input. Choosing good features is the most important skill in applied ML.

### What makes a feature good?

A good feature:
- Has a real causal relationship with what you're predicting
- Is available at prediction time
- Has enough variation to be useful

**Example from our project:** `rushing_YDS` is a good feature for RBs because players who rack up college rushing yards genuinely tend to be good NFL runners. The relationship is causal.

**Example of a bad feature we discovered:** `conference_Sun Belt` was a terrible feature. It had a strong statistical correlation in our training data (a few Sun Belt RBs happened to succeed), but no causal relationship — the conference itself doesn't make players better.

### Numeric vs Categorical Features

Features come in two flavors:

- **Numeric**: a number (rushing yards, touchdowns, fantasy points)
- **Categorical**: a category (position, conference)

The model can't directly use categories — it converts them into a series of 0/1 columns. "Position = RB" becomes a column called `position_RB` where 1 means yes and 0 means no. This is called **one-hot encoding**.

### Feature Engineering

Sometimes you derive new features from existing ones. We do this in two places:

- `total_touchdowns = passing_TD + rushing_TD + receiving_TD`
- `yards_from_scrimmage = rushing_YDS + receiving_YDS`

These derived features don't add new information the model doesn't already have, but they package it in a way that's easier to learn from. A model might struggle to learn "TDs from any source predict success" on its own, but if you hand it that combined number directly, it doesn't have to figure it out.

---

## 5. The Model: Random Forest

We use a **Random Forest**, which is one of the most reliable models for structured data (data in rows and columns, like ours).

### Decision Trees First

A Random Forest is built from many **decision trees**. A single decision tree is a series of if/then rules:

> If rushing_YDS > 800 AND rushing_TD > 8, predict 90 fantasy points.
> If receiving_REC > 70 AND receiving_TD > 5, predict 85 fantasy points.

Trees are easy to understand but they have a problem: they memorize the training data too well. A tree trained on our data might learn a rule that perfectly fits 2022 rookies but fails completely on 2026 rookies. This is called **overfitting**.

### Why "Forest"?

A Random Forest builds hundreds of these trees, each trained on a random sample of the data and a random subset of features. Then it averages their predictions.

This works because the trees' errors tend to cancel each other out. One tree might be fooled by a weird conference correlation; another won't have conference as a feature at all. The average is much more stable than any single tree.

### Preprocessing: Scaling

Before the Random Forest sees the data, we run it through a **StandardScaler**. This rescales every numeric feature to have a mean of 0 and a standard deviation of 1.

Without this, a feature like `passing_YDS` (values in the thousands) would visually dominate a feature like `receiving_TD` (values of 0-15), even though both could be equally important. Scaling puts them on equal footing.

---

## 6. Evaluation: How Do You Know If It's Working?

### Mean Squared Error (MSE)

The most basic way to measure a regression model: take every prediction, calculate how far off it was, square that error, and average them all.

Squaring matters because it punishes big mistakes more than small ones. Being off by 40 points is more than twice as bad as being off by 20 points.

Lower MSE = better model.

### R² (R-Squared)

R² is easier to interpret. It answers: "How much of the variation in rookie fantasy points does our model explain?"

- R² = 1.0: perfect predictions every time
- R² = 0.0: the model is no better than just predicting the average for everyone
- R² < 0: the model is actively worse than predicting the average

Our best no-pick model sits around **0.10**. That means we're explaining about 10% of the variance in rookie performance. That sounds low — and it is — but predicting rookie NFL performance from college stats alone is genuinely one of the hardest forecasting problems in sports. Even NFL scouts with years of experience and film study get this wrong constantly.

### Cross-Validation

You might think: train the model, test it on the training data, see how well it does. The problem is that a model can memorize the training data and score perfectly without actually learning anything useful. This is overfitting.

The fix is to test on data the model has never seen. We use **5-fold cross-validation**:

1. Split the data into 5 equal chunks
2. Train on chunks 1-4, test on chunk 5
3. Train on chunks 1-3 and 5, test on chunk 4
4. Repeat for all 5 combinations
5. Average the R² scores

This gives a much more honest picture of how the model will perform on truly new data — like 2026 rookies.

The ± number you see (e.g., `CV R² = 0.099 ± 0.163`) is the **standard deviation** across those 5 test folds. A high standard deviation means the model's performance varies a lot depending on which players end up in the test set. That's a sign the model is fragile.

---

## 7. Overfitting and Underfitting

These are the two failure modes of ML models.

### Overfitting

The model memorizes the training data instead of learning general patterns. Signs:
- Great performance on training data, terrible on new data
- SHAP values dominated by noisy features (conference, for example)

We saw this with the conference feature. The model learned "Sun Belt RBs score well" because a handful of Sun Belt RBs in our training data happened to be good. There weren't enough Sun Belt RBs in training for the model to learn a reliable pattern — it just memorized the coincidence.

Overfitting gets worse with more features relative to training rows. We have ~300-700 rows and 11+ features. That's a tight ratio.

### Underfitting

The opposite problem: the model is too simple to capture the real patterns. Signs:
- Poor performance on both training and new data
- Low R² everywhere

A constant model that always predicts "the average player scores 60 fantasy points" would be perfectly underfit.

### The Balance

More training data helps both problems. Adding 6 more years of training data (2014-2019) helped the with-pick model because draft position is a signal that's consistent across eras. It hurt the no-pick model because football changed enough that 2014 patterns actively misled the model.

---

## 8. SHAP Values: Opening the Black Box

Traditional feature importance tells you which features the model uses most. **SHAP values** (SHapley Additive exPlanations) tell you exactly how much each feature moved a specific prediction up or down.

For a player like Jeremiyah Love:
- `rushing_TD +13.31` means his rushing touchdowns pushed his predicted points 13 points higher than the baseline
- `passing_YPA -5.06` means his (zeroed) passing YPA pushed his prediction 5 points lower

SHAP is how we diagnosed problems in this project. Before we cleaned up the data, a Sun Belt RB named Zylan Perry had `conference_Sun Belt +17.31` as his #1 SHAP feature — meaning the model was ranking him almost entirely because of his conference, not his stats. That's a clear signal of a bad feature.

### The Passing YPA Artifact

We still see `passing_YPA -5` for non-QBs even after zeroing it out. This is a subtle artifact: the StandardScaler computes the average `passing_YPA` across all players including QBs (who have high values). Non-QBs get 0, which is well below that average, so the model sees it as a negative signal.

The fix would require separate models per position — but with only 40 QBs in training, that would make the QB model too unreliable. We accepted the artifact. It's symmetric across all non-QBs so it doesn't distort relative rankings within a position.

---

## 9. Two Models, Two Use Cases

We train two versions:

**No-pick model**: uses only college stats. Used right now, before the draft, to rank prospects.

**With-pick model**: adds draft pick number as a feature. Used after the April draft when we know where each player was selected.

Why does this matter? Draft position is one of the strongest predictors of NFL success — not because being picked early makes you better, but because NFL teams have more scouting information than we do. Their collective judgment, expressed through draft order, captures things our college stats miss: combine athleticism, injury history, scheme fit, character, football IQ.

The with-pick model has CV R² of 0.373, more than triple the no-pick model's 0.099. That gap tells you how much information is in draft position that our college stats don't capture.

---

## 10. What the Model Can't Know

Understanding a model's blind spots is as important as understanding what it does well.

**Age**: We tried to add this. The CollegeFootballData API doesn't expose birth dates, so we used "years of college experience" as a proxy. It didn't help because almost every prospect has 2-3 years of data — not enough variation to learn from.

**Scheme fit**: Blake Horvath (Navy) runs the triple-option. His college YPA looks great but it's a product of the system, not NFL-translatable skill. Our model sees high YPA and projects him as QB2. NFL analysts see "triple-option QB" and rank him outside the top 100.

**Opportunity**: A player landing on a bad team with an open starting spot will out-produce a better player buried on a depth chart. Our model has no idea where players will land.

**Physical traits**: Size, speed, injury history, athleticism — none of this is in our features.

**The comparison with FantasyPros**: Their top prospects (Carnell Tate, Jordyn Tyson, Makai Lemon) are consensus picks because of draft capital projections, age, and scouting grades — not college stats. Our model ranks them lower because their stat production was modest. Neither approach is wrong; they're measuring different things.

---

## 11. The Honest Assessment

Our model is most useful as a **sanity check and complement** to expert rankings, not a replacement.

Where it adds value:
- Identifying high-volume statistical producers who might be overlooked
- Flagging players whose rankings are being driven by noise (conference effects, system stats)
- Providing a baseline for comparison with consensus rankings

Where it falls short:
- Any player whose value depends on age, athleticism, or NFL landing spot
- QBs from non-traditional systems (option, RPO-heavy)
- Players with limited college exposure (1-2 seasons)

A CV R² of 0.10 is honest: the model explains about 10% of what makes rookies successful. The other 90% is things we can't measure, things that haven't happened yet, and genuine unpredictability. That's what makes it fun.

---

## 12. What Would Actually Improve This

In rough order of impact:

1. **Combine data** (40-yard dash, vertical, arm length): physical traits explain a lot of what college stats miss, and this data is publicly available after February
2. **Draft position** (already built, waiting for April draft): the biggest single predictor we don't have yet
3. **More recent training data**: as more seasons pass, the model will naturally improve
4. **Positional models with more data**: separate QB/RB/WR/TE models would be more accurate, but we need ~200+ rows per position to make them viable — probably a 2-3 year wait
5. **True age data**: birth year from a data source that exposes it would help identify breakout candidates vs older prospects

---

*Built with scikit-learn, pandas, and a lot of trial and error.*
