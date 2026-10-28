# Departures from your instruction

None. The five text changes listed here earlier follow notebook 14, as you said with "you should not have departure". The step 2 sentence of the configure skill is the monthly sentence again, with only the field name changed.

# Two daily skills carry notebook 14's synthetic historical control to the 52 treatment cases.

The skills are shc-configure and shc-design, each in its own folder of nine files under `notebooks/llm_band_sc/skills_daily`. In your words, "design is model design, config is her code model give config". The configure skill always fits with her code, and the model gives its settings. In the design skill the model designs the method, and a choice her code does not have runs our code, as in the monthly runs. This holds for the simple skills only, and the profile experiment defines its skills in its own way.

The model's working folder holds the visits given before 27 September 2024 and the dates of the visits to predict, between 27 and 64 per case, and no value after the visits given. The model predicts every band at every visit to predict. No model has been called.

# Four rows were settled after the parity report.

Your answer "if we can see her result, use her result. i don't care the 0.0001" settled two rows. Her reference row in every results table is the numbers notebook 14 printed, read from the notebook's saved outputs. A number she did not print, such as the counterfactual at each visit, comes from our run of her code. The parity gate counts a printed number that differs from hers by 0.0001 as reproduced.

Two rows follow from the monthly pipeline. Every run of her code uses one thread, as the sandbox the models use does. Her fit depends on the thread count, so with one thread everywhere the fitter in the sandbox at her settings reproduces our run of her code with difference 0. The code that runs a design choice her code does not have is the monthly code, and no gate runs it, as you asked with "you should not test with code deviate from her".

# Every gate of the plan passes.

| gate | what it checks | result |
|---|---|---|
| DK1 | the notebook 14 module is a byte copy of her notebook's statements | passes |
| DK2 | our run of her code prints her cell 5 and cell 8 lines exactly and her cell 12 lines within 0.0001 | passes at one thread; 13 of 309 cell 12 lines differ by 0.0001 |
| DK3 | her fit of the largest case, 242 visits given and 64 to predict, finishes inside the sandbox's command limit | 17.3 seconds of the 120 allowed |
| DK4 | the fitter at her settings reproduces our run of her code at every band and visit to predict of the 52 cases | passes with difference 0 at all 13,234 values |
| DK5 | every rule that ties the fields refuses a file that breaks it, and the configure skill refuses a design choice | passes |
| DK6 | the skill files match their manifest and the bundled modules match the package files | passes |
| DK7 | no value after the visits given reaches the working folder in a fit at her settings, and the sandbox hides the project, the home folder and the network | passes |
| DK8 | dry runs of both skills at her settings end with a valid answer equal to our run of her code | passes with difference 0 |
| DK9 | a command output of 20,000 characters reaches the model whole | passes |
| DK10 | the step cap, empty replies, empty output and resume behave as in the monthly runs | passes |


# The texts and one case's output are on a companion page.

Every text the model can read is shown beside its monthly text on `Docs/10-15-daily-skills-texts.md`, with the task text and the output of the two scripts for the Sentinel-1 parcel treatment_0001.


# The bundled modules are the package files byte for byte.

Each skill's scripts folder carries the three package modules below unchanged. The notebook 14 module is her notebook's code, extracted statement by statement.

| file in each skill's scripts folder | package file | sha256 |
|---|---|---|
| `lb_her_nb14.py` | `notebooks/llm_band_sc/lb_her_nb14.py` | `1bcbd1aa65cc0619fe0ef175d585c9ed53f9dc87552aa58833e9206a44e8f136` |
| `lb_spec_daily.py` | `notebooks/llm_band_sc/lb_spec_daily.py` | `d9351abf0bc62cbb2dcb04ac1e9d3dc14f03f5ebd35f7a988700b50f79ab048e` |
| `lb_fit_daily.py` | `notebooks/llm_band_sc/lb_fit_daily.py` | `c8d93c76413dee0c3fe8e3eb197d3018cce17167331e25f7bf5f292bd23632ec` |

