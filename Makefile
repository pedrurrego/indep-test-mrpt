# Independence test experiments
# Requires GNU Make 4+ (brew install make → gmake)
#
# Usage:
#   gmake run TEST=qad                  # one test, all generators (sequential)
#   gmake run TEST=qad -j4              # one test, all generators (4 in parallel)
#   gmake run GENERATOR=dep-sine-4pi      # all tests, one generator (sequential)
#   gmake run GENERATOR=dep-sine-4pi -j4  # all tests, one generator (4 in parallel)
#   gmake plot GENERATOR=dep-sine-4pi     # plot one generator's results
#   gmake plot-all                      # regenerate all charts from existing results
#
# Salahub quasi-power noise sweep (Zhang 2019 patterns):
#   gmake noise-run TEST=qad            # one test, all patterns (sequential)
#   gmake noise-run TEST=qad -j6        # one test, all patterns (6 in parallel)
#   gmake noise-run PATTERN=linear      # all tests, one pattern (sequential)
#   gmake noise-run PATTERN=linear -j4  # all tests, one pattern (4 in parallel)
#   gmake noise-plot PATTERN=linear     # one pattern's chart
#   gmake noise-plot-all                # regenerate every chart from result JSONs
#
# rBEX quasi-power resolution sweep (Zhang 2019 bisection expanding cross):
#   gmake rbex-run TEST=pug-grid TEST_CONFIG='{"norm":"density-multi"}'
#   gmake rbex-run -j4                  # every test across the resolution sweep
#   gmake rbex-plot                     # overlay chart of every result JSON
#   gmake rbex-plot-pattern             # rBEX scatterplot strip (sampler check)

CLI       := uv run python -m indep_tests
NOISE_CLI := uv run python -m indep_tests.salahub_noise
RBEX_CLI  := uv run python -m indep_tests.salahub_rbex

# --- Defaults (override on the command line) ---
B            ?= 500
R            ?= 100
ALPHA        ?= 0.05
SEED         ?= 1234
SAMPLE_SIZES ?= 20,50,100,200,500
TEST_CONFIG  ?= {}
COLLECT_P_VALUES ?=

# --- Noise-sweep defaults (follow Salahub & Oldford 2025 §5) ---
NOISE_N      ?= 500
NOISE_R      ?= 200
NOISE_B      ?= 1000
NOISE_ALPHA  ?= 0.01
NOISE_SEED   ?= 1234
NOISE_LEVELS ?=

# --- rBEX resolution-sweep defaults (Salahub 2025 §5.3) ---
RBEX_N           ?= 2000
RBEX_R           ?= 200
RBEX_B           ?= 500
RBEX_ALPHA       ?= 0.01
RBEX_SEED        ?= 1234
RBEX_RESOLUTIONS ?=

GENERATORS := $(shell $(CLI) list-generators | tail -n +2 | tr -d ' ')
TESTS      := $(shell $(CLI) list-tests      | tail -n +2 | tr -d ' ')
PATTERNS   := $(shell $(NOISE_CLI) list-patterns | tail -n +2 | tr -d ' ')

RESULTS_DIR       := indep_tests/results
NOISE_RESULTS_DIR := indep_tests/salahub_noise/results
NOISE_CHARTS_DIR  := $(NOISE_RESULTS_DIR)/charts
RBEX_RESULTS_DIR  := indep_tests/salahub_rbex/results
RBEX_CHARTS_DIR   := $(RBEX_RESULTS_DIR)/charts

# -- Targets for parallelism --------------------------------------
#    Each combination becomes its own target so `gmake -j` works.

RUN_GEN_TARGETS  := $(addprefix run-gen-,$(GENERATORS))
RUN_TEST_TARGETS := $(addprefix run-test-,$(TESTS))

.PHONY: run $(RUN_GEN_TARGETS) $(RUN_TEST_TARGETS) plot plot-all

ifdef TEST
# gmake run TEST=qad  →  one test, all generators
run: $(RUN_GEN_TARGETS)
else ifdef GENERATOR
# gmake run GENERATOR=dep-sine-4pi  →  all tests, one generator
run: $(RUN_TEST_TARGETS)
else
run:
	@echo "ERROR: TEST or GENERATOR is required."
	@echo "  gmake run TEST=qad                 # one test, all generators"
	@echo "  gmake run GENERATOR=dep-sine-4pi     # all tests, one generator"
	@exit 1
endif

$(RUN_GEN_TARGETS): run-gen-%:
	$(CLI) run \
		--test $(TEST) \
		--generator $* \
		--sample-sizes $(SAMPLE_SIZES) \
		--alpha $(ALPHA) \
		--r $(R) \
		--b $(B) \
		--seed $(SEED) \
		--test-config '$(TEST_CONFIG)' \
		$(if $(COLLECT_P_VALUES),--collect-p-values) \
		--output $(RESULTS_DIR)

$(RUN_TEST_TARGETS): run-test-%:
	$(CLI) run \
		--test $* \
		--generator $(GENERATOR) \
		--sample-sizes $(SAMPLE_SIZES) \
		--alpha $(ALPHA) \
		--r $(R) \
		--b $(B) \
		--seed $(SEED) \
		--test-config '$(TEST_CONFIG)' \
		$(if $(COLLECT_P_VALUES),--collect-p-values) \
		--output $(RESULTS_DIR)

# -- Plot charts ---------------------------------------------------

plot:
ifndef GENERATOR
	@echo "ERROR: GENERATOR is required."
	@echo "  gmake plot GENERATOR=dep-sine-4pi"
	@exit 1
endif
	$(CLI) plot --generator $(GENERATOR) --output $(RESULTS_DIR)/charts

# -- Regenerate all charts -----------------------------------------
#    Discovers generators from result filenames in RESULTS_DIR.

plot-all:
	@for f in $(RESULTS_DIR)/*.json; do \
		basename "$$f" | cut -d_ -f1; \
	done | sort -u | while read -r gen; do \
		echo "Plotting $$gen ..."; \
		$(CLI) plot --generator "$$gen" --output $(RESULTS_DIR)/charts; \
	done

# -- Salahub quasi-power noise sweep -------------------------------

NOISE_RUN_PAT_TARGETS  := $(addprefix noise-run-pat-,$(PATTERNS))
NOISE_RUN_TEST_TARGETS := $(addprefix noise-run-test-,$(TESTS))

.PHONY: noise-run $(NOISE_RUN_PAT_TARGETS) $(NOISE_RUN_TEST_TARGETS) noise-plot noise-plot-all

ifdef TEST
# gmake noise-run TEST=qad  →  one test, all patterns
noise-run: $(NOISE_RUN_PAT_TARGETS)
else ifdef PATTERN
# gmake noise-run PATTERN=linear  →  all tests, one pattern
noise-run: $(NOISE_RUN_TEST_TARGETS)
else
noise-run:
	@echo "ERROR: TEST or PATTERN is required."
	@echo "  gmake noise-run TEST=qad          # one test, all patterns"
	@echo "  gmake noise-run PATTERN=linear    # all tests, one pattern"
	@exit 1
endif

$(NOISE_RUN_PAT_TARGETS): noise-run-pat-%:
	$(NOISE_CLI) run \
		--test $(TEST) \
		--pattern $* \
		--n $(NOISE_N) \
		--r $(NOISE_R) \
		--b $(NOISE_B) \
		--alpha $(NOISE_ALPHA) \
		--seed $(NOISE_SEED) \
		$(if $(NOISE_LEVELS),--levels $(NOISE_LEVELS)) \
		--test-config '$(TEST_CONFIG)' \
		--output $(NOISE_RESULTS_DIR)

$(NOISE_RUN_TEST_TARGETS): noise-run-test-%:
	$(NOISE_CLI) run \
		--test $* \
		--pattern $(PATTERN) \
		--n $(NOISE_N) \
		--r $(NOISE_R) \
		--b $(NOISE_B) \
		--alpha $(NOISE_ALPHA) \
		--seed $(NOISE_SEED) \
		$(if $(NOISE_LEVELS),--levels $(NOISE_LEVELS)) \
		--test-config '$(TEST_CONFIG)' \
		--output $(NOISE_RESULTS_DIR)

noise-plot:
ifndef PATTERN
	@echo "ERROR: PATTERN is required."
	@echo "  gmake noise-plot PATTERN=linear"
	@exit 1
endif
	$(NOISE_CLI) plot \
		--pattern $(PATTERN) \
		--alpha $(NOISE_ALPHA) \
		--results-dir $(NOISE_RESULTS_DIR) \
		--output-dir $(NOISE_CHARTS_DIR)

noise-plot-all:
	$(NOISE_CLI) plot-all \
		--alpha $(NOISE_ALPHA) \
		--results-dir $(NOISE_RESULTS_DIR) \
		--output-dir $(NOISE_CHARTS_DIR)

# -- rBEX quasi-power resolution sweep ----------------------------
#    Only one fan-out axis (the test); the pattern is always rBEX.

RBEX_RUN_TEST_TARGETS := $(addprefix rbex-run-test-,$(TESTS))

.PHONY: rbex-run $(RBEX_RUN_TEST_TARGETS) rbex-plot rbex-plot-pattern

ifdef TEST
# gmake rbex-run TEST=pug-grid  →  one test across the resolution sweep
rbex-run: rbex-run-test-$(TEST)
else
# gmake rbex-run [-jN]  →  every test across the resolution sweep
rbex-run: $(RBEX_RUN_TEST_TARGETS)
endif

$(RBEX_RUN_TEST_TARGETS): rbex-run-test-%:
	$(RBEX_CLI) run \
		--test $* \
		--n $(RBEX_N) \
		--r $(RBEX_R) \
		--b $(RBEX_B) \
		--alpha $(RBEX_ALPHA) \
		--seed $(RBEX_SEED) \
		$(if $(RBEX_RESOLUTIONS),--resolutions $(RBEX_RESOLUTIONS)) \
		--test-config '$(TEST_CONFIG)' \
		--output $(RBEX_RESULTS_DIR)

rbex-plot:
	$(RBEX_CLI) plot \
		--alpha $(RBEX_ALPHA) \
		--results-dir $(RBEX_RESULTS_DIR) \
		--output-dir $(RBEX_CHARTS_DIR)

rbex-plot-pattern:
	$(RBEX_CLI) plot-pattern \
		--n $(RBEX_N) \
		$(if $(RBEX_RESOLUTIONS),--resolutions $(RBEX_RESOLUTIONS)) \
		--output-dir $(RBEX_CHARTS_DIR)
