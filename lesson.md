# Lesson — L01 Introduction to Machine Learning

> *Sarah Chen · Customer Experience Analyst · NorthStar Retail · January 2023.*
> Her first Monday. Aisha from Customer Service hands her a USB drive with 10,000 reviews. Priya the CMO wants sentiment and topic breakdown by Friday.

Use this document as your concept reference — before, during, and after the session. Each section explains a key idea in plain English, anchors it to Sarah's scenario at NorthStar, and shows why it matters for the rest of the course.

| Section | Notebook | Time |
|---|---|---|
| Part 1: What is ML? | `notebooks/02_what_is_ml.ipynb` | ~30 min |
| Part 2: Three categories of ML | `notebooks/03_three_categories.ipynb` | ~30 min |
| Part 3: The ML workflow | `notebooks/04_ml_workflow.ipynb` | ~30 min |

---

## Why Machine Learning matters

Every business generates data — customer interactions, sales, support tickets, sensor readings, photos, documents. Traditional software turns that data into decisions by following rules a programmer writes. Machine Learning turns that data into decisions by **letting the computer discover the rules itself** from examples.

This matters for three reasons:

1. **Some problems have no clean rules.** Nobody can write a rule that defines a "cat" pixel-by-pixel or a "positive review" word-by-word, but millions of labelled examples exist.
2. **Data is cheaper than expertise.** A company often has more data than it has experts to hand-write rules for.
3. **Patterns change over time.** A rule-based fraud detector is outdated the moment fraudsters change tactics; an ML model retrained on new data keeps up.

You will see all three reasons play out in Sarah's scenario at NorthStar Retail.

---

## Part 1: What is Machine Learning?

### The core idea — two kinds of programming

**The idea in plain English:** In traditional programming, a person writes rules and the computer follows them. In Machine Learning, a person provides examples (inputs and correct outputs) and the computer figures out the rules by itself.

**Real-world analogy:** Imagine teaching a child the difference between a dog and a cat. You could try to write a rulebook: "dogs have floppy ears, cats have pointy ears, dogs bark..." You'd fail — there are too many exceptions. Instead, you show the child many dogs and many cats, each labelled, and their brain builds its own sense of what's what. That's Machine Learning.

**Why it matters:** Every single method we study in this course (regression, trees, neural networks, GenAI) is a different way of "showing examples to the computer and letting it learn the rules." Understanding this frame keeps the course from feeling like a pile of unrelated tricks.

---

### Key vocabulary

| Term | Plain-English meaning |
|---|---|
| **Features** | The *inputs* to the model — whatever you know about a situation. For a customer review, features are the words; for a loan application, features are age, income, employment length. |
| **Label** | The *answer* — what we want the model to predict. For reviews, labels are "positive" / "negative"; for loans, "defaulted" / "repaid." |
| **Model** | The learned "rules" that map features to a prediction. You can think of it as a function: `prediction = model(features)`. |
| **Training** | The process of showing labelled examples to an algorithm so it builds the model. |
| **Inference** | Using a trained model on new, unlabelled data. (Sarah runs inference when she applies the model to her 10,000 unclassified reviews.) |
| **Pre-trained model** | A model someone else trained on a huge dataset. You can use it directly without doing the training yourself. This is what Sarah uses in class. |

---

### When is ML the right tool?

ML is the right tool when **all three** of these are true:

1. **The rules are hard to write by hand** (not a formula, not a policy you can encode).
2. **You have data with examples** of inputs and correct outputs — or you can get it.
3. **Being right most of the time is good enough** — ML is probabilistic, not deterministic.

ML is the **wrong** tool when:

- The rules are obvious and exact (tax calculations, unit conversions).
- The cost of any wrong answer is catastrophic (flight-control software).
- You have no data. ML without data is astrology with more maths.

---

### Quick Check — Part 1

**Q1:** Sarah's company also wants to automatically send a coupon to every customer whose birthday is today. Rule-based or ML?

> **Sample answer:** Rule-based. "Is today equal to their birthday?" is a one-line `if` statement. No data needed, no uncertainty. An ML model here would add cost and risk for zero benefit.

**Q2:** A bank wants to detect fraud in real-time transactions. They have millions of historical transactions labelled fraudulent / not-fraudulent. Which requirement for "ML is the right tool" do they satisfy, and which one might be tricky?

> **Sample answer:** They satisfy #1 (fraud patterns are too varied to hand-code) and #2 (lots of labelled data). Requirement #3 is tricky: a false negative (missing real fraud) is expensive but tolerable; a false positive (blocking a legitimate customer) is annoying but recoverable. They need to weigh the cost of each kind of wrong answer — a classic ML design decision we revisit in Lesson 3.

**Q3:** What's the difference between training and inference?

> **Sample answer:** Training is the (often slow, compute-heavy) process of learning the rules from labelled examples. Inference is using the trained model to predict on new data — it's what you do every time the model gives you an answer. In Sarah's case, someone else already trained the sentiment model; she only does inference.

---

## Part 2: The three categories of ML

Every ML problem fits into one of three broad categories — distinguished by **what data you have** and **what you want the model to do**.

### Supervised Learning

**The idea in plain English:** You have input-output pairs (features + label), and you want the model to learn the mapping so it can predict labels for new, unlabelled inputs.

**Real-world analogy:** A student studying past exam papers with answers. They see many questions and the correct answers, and over time, they learn to answer new questions on their own.

**Why it matters:** Supervised learning is **the workhorse of business ML**. Churn prediction, fraud detection, demand forecasting, sentiment analysis, loan approval — all supervised. Lessons L03, L04, and L05 cover this in depth.

**When to use:** You have labels. You want predictions.

---

### Unsupervised Learning

**The idea in plain English:** You have input data but no labels. You want the model to find **structure** in the data — groups, patterns, anomalies — on its own.

**Real-world analogy:** A librarian faced with a crate of new books, none labelled by genre. She groups them by "these feel similar" — romance over here, mystery over there — even though no one told her what the genres are.

**Why it matters:** Unsupervised learning shines for **exploration**: customer segmentation, anomaly detection in logs, document clustering, topic discovery. Lesson L06 covers this.

**When to use:** You don't have labels — or getting labels is too expensive — and you want to understand what's in the data.

---

### Reinforcement Learning

**The idea in plain English:** An agent learns by interacting with an environment: it tries actions, gets rewards or penalties, and adjusts its behaviour over time to maximise reward.

**Real-world analogy:** Training a dog with treats. Sit → treat. Bark at the mailman → no treat. The dog learns which actions pay off.

**Why it matters:** Reinforcement learning is the engine behind game-playing AIs, robotic control, ad bidding, and parts of ChatGPT's training. **We will not build RL systems in this course** — it is a big field and beyond the 10-lesson scope — but you should recognise the category when it appears in the real world.

**When to use:** You can't list correct answers in advance — you can only give the system *feedback on what it does*, and you need it to explore.

---

### Comparison at a glance

| Dimension | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| **Data you need** | Features + labels | Features only | Environment + reward signal |
| **Goal** | Predict labels | Find structure | Maximise reward over time |
| **Example** | Churn prediction | Customer segmentation | Game-playing AI |
| **Covered in** | L03, L04, L05, L08, L09 | L06 | Mentioned, not built |

---

### Quick Check — Part 2

**Q1:** Which category does Sarah's review-classification problem fit into, given she has a training set of 50,000 past reviews already labelled positive or negative?

> **Sample answer:** Supervised learning. She has features (the review text) and labels (positive/negative). She wants the model to predict labels for new reviews.

**Q2:** A newspaper wants to automatically group its last 10 years of articles into "topics" — without specifying the topics in advance. Which category?

> **Sample answer:** Unsupervised. No labels. The goal is to discover structure (what topics exist).

**Q3:** Could Sarah's problem also be tackled with unsupervised learning? What would be different?

> **Sample answer:** Yes — she could cluster the 10,000 reviews into, say, 5 groups based on how similar they are. She wouldn't know which group is "positive" or "about sizing" without inspecting a sample. It would be faster to set up (no labels needed) but less useful to her manager, who asked specific labelled questions.

---

## Part 3: The ML workflow

An ML project is not "train a model." It's a process with seven steps, and most of the work is **before and after** the training.

### The 7 steps

1. **Frame the problem.** What is the business question? What will someone *do* with the answer? What counts as success?
2. **Collect the data.** Where does it live? Is it enough? Is it representative?
3. **Clean and explore.** Missing values, duplicates, wrong types, biased samples.
4. **Train the model.** Choose a method, fit it to the data.
5. **Evaluate on unseen data.** Is it good enough on examples it hasn't seen?
6. **Deploy.** Make the model available to the people or systems that will use it.
7. **Monitor.** The world changes; the model decays. Track its performance over time.

### Common time split

On a real project, the time distribution often looks like:

- Framing + collection + cleaning: **60–70%**
- Training + evaluation: **10–20%**
- Deployment + monitoring: **20%**

**Beginners often imagine the split is the opposite.** Understanding this split is part of becoming a practitioner.

---

### Framing: the most important step

**The idea in plain English:** Before you write any code, get specific about:

- What is the **question**? ("Will this customer cancel?")
- Who will **act** on the answer? (Retention team, marketing automation, nobody?)
- What is **success**? (90% accuracy? Saving $100k in churn? Customer delight?)
- What is the **cost of being wrong**? (Both kinds — false positives and false negatives.)

**Real-world analogy:** Before a detective starts searching a house, they ask: *"What am I actually looking for? Who will use the evidence? What happens if I find nothing?"* ML projects without framing are detectives searching without a theory.

**Why it matters:** A beautifully trained model that solves the wrong problem is worth zero. Sarah's task seems obvious ("classify reviews") until you ask: *positive/negative only? or also topic? whose definition of "positive"? and what happens after she labels them?*

---

### Evaluation: keeping yourself honest

**The idea in plain English:** A model that scores 99% on the data it was trained on might score 60% on data it has never seen. You must hold out some data as a test set and only score the model on that.

**Real-world analogy:** A student who memorised the answers to last year's exam might score 100% on that exam — but tell you nothing about whether they understand the material. The real test is a new exam.

**Why it matters:** Every model evaluation in this course — and every serious ML project in the world — uses a train/test split (or better, cross-validation, which we'll cover in L04).

---

### Deployment and monitoring — the hidden 40%

**The idea in plain English:** Once the model works in a notebook, you have to (a) make it runnable in production (API, batch pipeline, embedded), (b) track how it's doing in the wild, and (c) decide when to retrain.

**Real-world analogy:** Shipping a product is different from designing a prototype. A prototype that works on the lab bench may fall apart when actual customers use it. Same for models.

**Why it matters:** Many beginner projects stop at Step 5 (evaluation on a notebook). Senior practitioners are distinguished by how seriously they take Steps 6 and 7.

---

### Quick Check — Part 3

**Q1:** Sarah has been asked to classify reviews by Friday. Her manager says "90% accuracy is fine." Is the framing complete?

> **Sample answer:** No. Open questions:
>
> - *Accuracy on what?* Overall, or on the rare "severe complaint" cases specifically?
> - *Who acts on it?* If an angry-customer coupon is auto-sent, a false positive costs real money.
> - *What about topic tagging?* The manager also wanted sizing / quality / delivery — a separate problem.
>
> Push back is part of the job. Reframing the question is the work.

**Q2:** A model scored 98% on the training set and 70% on the held-out test set. What's going on and what would you do?

> **Sample answer:** Classic **overfitting** — the model memorised the training data instead of learning general patterns. You'd try a simpler model, more regularisation, more training data, or fewer/better features. We'll cover overfitting properly in L03 and L04.

**Q3:** You deploy a fraud detection model. Six months later accuracy has dropped 10%. What happened, and which step of the workflow failed?

> **Sample answer:** Likely **data drift** — fraudsters changed tactics, or normal customer behaviour shifted. Step 7 (monitoring) caught the decay; Step 4 (training) or the data pipeline needs to be rerun on more recent data. This is why "train once and forget" doesn't work in practice.

---

## Putting it all together

A real ML practitioner's day is not "training models." It's **choosing** whether ML is the right tool in the first place, **framing** the problem to ensure the model will be used, **preparing** the data (the bulk of the work), and **monitoring** what happens once the model is live.

You will see this pattern repeat through every lesson in this course. We'll go deeper on supervised methods in L03–L05, unsupervised in L06, deep learning in L07, vision and language applications in L08 and L09, and modern GenAI in L10 — but every lesson sits inside this same outer workflow.

---

## Sarah's unanswered question (the bridge to L02)

By the end of class, Sarah has classified 10,000 reviews in minutes. But Priya is quiet for a moment and then asks:

> *"Sarah — your model says the majority of reviews are positive. But are the positive ones actually positive? How do we know we can trust that number?"*

Sarah doesn't have an answer today. That question — *how sure are we?* — is the engine of Lesson 2: Probability and Statistics. Come to L02 ready to help her answer it.

---

**Before you move on,** work through the three in-class notebooks (`02_what_is_ml.ipynb`, `03_three_categories.ipynb`, `04_ml_workflow.ipynb`), then attempt the assignment (Sarah goes on secondment to Lakeside Bank).
