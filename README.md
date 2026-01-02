# Imperial_ML_Module12_BO
**Module 12 Capstone – Final Consolidated Reflection (Functions 1–8)**
**What this capstone is really about (my understanding)**
At the start of this capstone, my first reaction was simple and honest:
“Why can’t I just sort the outputs in descending order and pick the best?”
With only a few data points and expensive evaluations, that felt like common sense.
As I progressed from Function 1 to the later high-dimensional problems (especially the cake recipe and hyperparameter tuning), I realised why that thinking breaks down. This capstone is not about finding a perfect answer; it is about making the most defensible next decision under uncertainty.
That is where Bayesian Optimisation (BO) actually earns its place.

**How I worked (common approach for all functions)**

For every function, I followed the same structured process, but relied on different parts of it depending on complexity:
    1. Look at the raw data first
Understand the scale, sign, and spread of outputs (especially important when values are near-zero or all negative).
    2. Visualise before modelling
I used a reusable pairwise plotting function to generate all 2D projections.
This helped me spot:
        ◦ Flat landscapes
        ◦ Noise-driven outliers
        ◦ Regions that looked consistently “less bad” or promising
    3. Challenge simple ranking
I actively tested whether descending order actually made sense.
In most cases beyond 2D, it didn’t.
    4. Introduce Bayesian Optimisation when intuition failed
I used a Gaussian Process + UCB acquisition to:
        ◦ Balance exploration and exploitation
        ◦ Avoid over-committing to noisy extremes
        ◦ Justify the next query mathematically
The key shift was moving from “what looks best” to “what is the safest next query to learn from.”
Function-wise summary (how I derived the final values)

**Function 1 (2D – flat, near-zero outputs)**
Final: 0.890000-0.650000
Almost all outputs were extremely close to zero, with one strong negative outlier. Sorting values was useless. Visualisation showed a narrow region of “least bad” points, so I prioritised exploration and stability rather than magnitude.
**Function 2 (2D – noisy but structured)**
Final: 0.900000-0.700000
Here, descending order appeared to work because higher outputs clustered. However, noise was present. I used BO to confirm that the visually promising region was not just a coincidence.
**Function 3 (3D – weak visual signal)**
Final: 0.520000-0.650000-0.300000
Pairwise plots were inconclusive and no single dimension dominated. This is where visual intuition stopped helping. BO was used to select a balanced point supported by uncertainty estimates rather than raw output.
**Function 4 (4D – interacting variables)**
Final: 0.590000-0.420000-0.410000-0.230000
With four dimensions, interactions mattered more than individual values. I used all pairwise projections to narrow down reasonable ranges, then refined the decision using BO to account for cross-dimensional effects.
**Function 5 (4D – unimodal chemical yield)**
Final: 0.220000-0.850000-0.800000-0.900000
Outputs had a very wide range and extreme values dominated rankings. Since the function is described as unimodal, chasing extremes felt risky. BO helped focus around a stable peak instead of noisy outliers.
**Function 6 (5D – cake recipe, negative scores)**
Final: 0.224927-0.302504-0.431427-0.766288-0.080674
This function changed my mindset completely. All scores were negative, so “best” meant closest to zero, not highest. Descending order was meaningless. BO reframed the task as systematically reducing penalty while balancing ingredient trade-offs.
**Function 7 (6D – ML hyperparameter tuning)**
Final: 0.073653-0.456823-0.216125-0.189332-0.423049-0.851392
At six dimensions, intuition and ranking both failed. The space was sparse and noisy. I leaned heavily into exploration using BO, treating this as a realistic hyperparameter optimisation problem rather than a search for a single magic point.
**Function 8 (8D – highly complex black-box)**
Final:
0.032321-0.168323-0.317887-0.007847-0.789622-0.742835-0.133748-0.502796

At eight dimensions, global optimisation is unrealistic with limited data. The goal shifted to finding a strong local maximum. BO was the only practical way to manage uncertainty and propose a defensible next query.
**Final reflection (what actually changed in my thinking)**
I started this capstone thinking: “Just sort the outputs.”
**I finished it understanding:**
“Sorting ignores uncertainty, sparsity, noise, and interactions.”
Bayesian Optimisation became necessary not because it is sophisticated, but because it is the only method that scales decision-making as complexity increases. My objective was never to guarantee the best possible answer, but to make the best justified next move given limited information.
