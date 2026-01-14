Minimal Proto-Intelligence from Complexity

This repository contains a single Python script (proto\_intelligence.py) that runs a fixed set of deterministic simulations of a 32×32 2D grid dynamical system (toroidal wrap boundary), computes frozen metrics, writes CSV outputs, generates figures strictly from the CSV files (not from in-memory arrays), and writes a JSON runlog capturing parameters, seeds, and environment versions.

Requirements

*   Python 3
    
*   numpy
    
*   matplotlib
    

Install dependencies (example)pip install numpy matplotlib

How to runFrom the repo directory:python proto\_intelligence.py

You will be prompted:

*   \[1\] Quick mode — pipeline check
    
*   \[2\] Full mode — paper results
    

Modes

*   Quick mode: exactly 6 simulations total(1 A/B pair for each condition: full, no\_feedback, no\_nonlinearity)
    
*   Full mode: exactly 54 simulations total(27 A/B pairs distributed across main + robustness blocks)Both modes use the same frozen model parameters, grid size, and time indexing.
    

Outputs (auto-created folders)Each execution creates a RUN\_TAG and writes files into:

*   results/
    
*   figures/
    
*   runlog/
    

CSV outputs

*   results/master\_metrics\_.csvPer-run frozen metrics, including history dependence (A/B separation) at τ ∈ {10, 50, 150}.
    
*   results/timeseries\_.csvPer-timestep rows for every simulation run (t = 0..359), including:
    
    *   global\_stat(t) = mean(abs(fast(t)))
        
    *   pred\_error(t) = mean(abs(slow\_prev - fast\_next)), computed immediately after the fast update (before perturbation and before slow update)
        

Figures (generated from CSV only)

*   figures/fig1\_model\_schematic\_.png
    
*   figures/fig2\_time\_evolution\_.png
    
*   figures/fig3\_history\_dependence\_.png
    
*   figures/fig4\_ablation\_collapse\_.png
    
*   figures/fig5\_self\_prediction\_.png
    
*   figures/fig6\_robustness\_.png (optional; code path exists but default disabled)
    

Runlog JSON

*   runlog/run\_.jsonIncludes:
    
*   run tag, mode, master seed, and all derived per-run seeds
    
*   frozen parameters (GRID, STEPS, phases, clipping, alpha/beta, perturbation settings, boundary)
    
*   explicit time indexing statement (warmup/perturb/observe ranges and t\_end)
    
*   Python / numpy / matplotlib versions and timestamp
    

Conditions (frozen)Required conditions:

1.  full
    
2.  no\_feedback (sets beta = 0.0)
    
3.  no\_nonlinearity (replaces tanh with identity; clipping still applied)
    

An optional condition frozen\_slow exists in code only behind a disabled constant flag by default.

Important note (scope)This implementation is frozen by specification and explicitly does not implement:

*   goals
    
*   learning
    
*   an environment
    
*   policies
    
*   rewards
    
*   agents
    
*   messaging
    

Memory is implicit via recursion in the state updates; there are no history buffers or recall mechanisms.